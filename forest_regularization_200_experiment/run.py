"""Isolated, resumable 200-tree forest regularization experiment."""

import argparse
import gc
import hashlib
import json
import os
import re
import sys
import threading
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
HELPERS = REPO / "backend/forest_regularization_experiment"
sys.path[:0] = [str(REPO / "backend/src"), str(HELPERS)]
os.environ["NUMBA_CACHE_DIR"] = str(ROOT / "runtime/numba")
os.environ["MPLCONFIGDIR"] = str(ROOT / "runtime/matplotlib")

import joblib
import numpy as np
import pandas as pd
from compare_cutoffs import metric_row, verify_grid
from evaluate_csv import check_counts, load_csv
from experiment import (
    KINDS,
    bundle_inventory,
    make_cohorts,
    predict,
    read_json,
    sample_positions,
    time_summary,
    time_sweep,
    verify_reuse,
    verify_unchanged,
)
from experiment import shap_values as explain_forest
from integritree.config import ExperimentConfig
from integritree.ml.artifacts import PACKAGES, load_bundle
from integritree.ml.data import file_sha256
from integritree.ml.features import FEATURE_COLUMNS
from integritree.ml.inference import feature_matrix
from integritree.ml.preparation import load_prepared_split
from integritree.ml.preprocessing import FittedPreprocessor
from integritree.ml.research import new_run, verify_prepared
from integritree.ml.threshold_search import percent_grid_thresholds
from integritree.ml.training import (
    forest_parameters,
    memory_preflight,
    resample_training,
)
from sklearn.ensemble import RandomForestClassifier

GRID = [[20, 1], [16, 1], [12, 1], [20, 10], [20, 50], [16, 10]]


def shap_values(model, background, values):
    # sklearn evaluates float32 inputs. The SHAP C kernel and expected-value
    # calculation must see the same rounded background at split boundaries.
    return explain_forest(model, background.astype(np.float32), values)


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def signature(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def files_in(folder):
    return {
        p.relative_to(folder).as_posix(): file_sha256(p)
        for p in folder.rglob("*")
        if p.is_file()
        and p.name != "checkpoint.json"
        and not p.name.endswith(".partial")
    }


def validate_files(folder, inventory):
    for name, expected in inventory.items():
        path = (folder / name).resolve()
        if not path.is_relative_to(folder.resolve()) or file_sha256(path) != expected:
            raise ValueError(f"Checkpoint file mismatch: {name}")


def read_config(path):
    settings = read_json(path)
    expected = {
        "schema_version",
        "saved_candidates",
        "prepared",
        "validation_csv",
        "previous_results",
        "n_estimators",
        "forests",
        "seed",
        "background_rows",
        "shap_uniform_rows",
        "shap_fraud_rows",
        "shap_baseline_flagged_rows",
        "sensitivity_rows",
    }
    if (
        set(settings) != expected
        or settings["schema_version"] != 1
        or settings["forests"] != GRID
    ):
        raise ValueError("Use the approved six-configuration protocol")
    for name, value in {
        "n_estimators": 200,
        "seed": 42,
        "background_rows": 200,
        "shap_uniform_rows": 200,
        "shap_fraud_rows": 100,
        "shap_baseline_flagged_rows": 50,
        "sensitivity_rows": 50,
    }.items():
        if type(settings[name]) is not int or settings[name] != value:
            raise ValueError(f"Unsupported {name}")
    for name in ("saved_candidates", "prepared", "validation_csv", "previous_results"):
        settings[name] = str((path.parent / settings[name]).resolve())
    return settings


def candidate_name(depth, leaf):
    return f"trees_200_depth_{depth}_leaf_{leaf}"


def inspect_sources(settings):
    base = Path(settings["saved_candidates"])
    sources = {}
    for leaf in (1, 10, 50):
        parent = base / f"ratio_0.01_trees_200_depth_20_leaf_{leaf}"
        options = [
            p.parent
            for p in parent.glob("*/metadata.json")
            if read_json(p).get("status") == "complete"
        ]
        if len(options) != 1:
            raise ValueError(f"Expected exactly one saved pair: {parent}")
        sources[leaf] = options[0]
    reference = load_bundle(sources[1])
    config = reference.config
    expected = forest_parameters(config, jobs=1)
    if (
        config.random_forest.n_estimators,
        config.random_forest.max_depth,
        config.random_forest.min_samples_leaf,
        config.smote.method,
        config.smote.sampling_ratio,
        config.smote.k_neighbors,
        config.seeds.model,
        config.seeds.smote,
    ) != (200, 20, 1, "smote_encoded", 0.01, 5, 42, 42):
        raise ValueError("Baseline training protocol mismatch")
    prepared = Path(settings["prepared"])
    prep = verify_prepared(reference, prepared)
    inventory = {}
    for leaf, source in sources.items():
        inventory.update(bundle_inventory(source))
        if leaf != 1:
            other = load_bundle(source)
            verify_reuse(other, reference, expected | {"min_samples_leaf": leaf})
            del other
    for name in (
        "metadata.json",
        "configuration.json",
        "preprocessing.json",
        "split_manifest.parquet",
        "train_features.parquet",
    ):
        path = prepared / name
        digest = file_sha256(path)
        if name != "metadata.json" and digest != prep["files"][name]:
            raise ValueError(f"Prepared fingerprint mismatch: {name}")
        inventory[str(path)] = digest
    csv_path = Path(settings["validation_csv"])
    inventory[str(csv_path)] = file_sha256(csv_path)
    previous = Path(settings["previous_results"])
    previous_manifest = read_json(previous / "manifest.json")
    if previous_manifest["status"] != "complete":
        raise ValueError("Previous reference experiment is incomplete")
    inventory[str(previous / "manifest.json")] = file_sha256(previous / "manifest.json")
    for name in ("comparison.csv", "time_shap.csv"):
        path = previous / name
        inventory[str(path)] = file_sha256(path)
        if inventory[str(path)] != previous_manifest["files"][name]:
            raise ValueError("Previous reference results are corrupted")
    plan = [
        {
            "candidate": candidate_name(d, leaf),
            "max_depth": d,
            "min_samples_leaf": leaf,
            "source": str(sources[leaf]) if d == 20 else None,
            "new_fits": 0 if d == 20 else 2,
        }
        for d, leaf in GRID
    ]
    result = config.model_dump(), plan, inventory, prep
    del reference
    gc.collect()
    return result


def training_context(config, inventory, parameters):
    # CSV/SHAP changes must not alter a training artifact's identity.
    paths = {
        p: digest
        for p, digest in inventory.items()
        if Path(p).name
        in ("train_features.parquet", "split_manifest.parquet", "preprocessing.json")
    }
    return signature(
        {"config": config.model_dump(), "inputs": paths, "parameters": parameters}
    )


def checkpoint_model(folder, kind, context):
    marker = folder / f"{kind}_checkpoint.json"
    if not marker.exists():
        return None
    data = read_json(marker)
    path = folder / f"{kind}.joblib"
    if data["context"] != context or data["sha256"] != file_sha256(path):
        raise ValueError(f"Incompatible or corrupted trained-model checkpoint: {path}")
    return data


def fit_one(folder, kind, values, labels, parameters, context):
    folder.mkdir(parents=True, exist_ok=True)
    completed = checkpoint_model(folder, kind, context)
    if completed is not None:
        return completed
    started = time.monotonic()
    model = RandomForestClassifier(**parameters).fit(values, labels)
    expected = predict(model, values[:128])
    path = folder / f"{kind}.joblib"
    partial = folder / f"{kind}.joblib.partial"
    joblib.dump(model, partial, compress=3)
    del model
    gc.collect()
    model = joblib.load(partial)
    np.testing.assert_allclose(
        predict(model, values[:128]), expected, atol=1e-12, rtol=0
    )
    if any(model.get_params()[key] != value for key, value in parameters.items()):
        raise ValueError("Reloaded training parameters differ")
    partial.replace(path)
    data = {
        "context": context,
        "sha256": file_sha256(path),
        "parameters": parameters,
        "rows": len(labels),
        "fraud_rows": int(np.sum(labels)),
        "elapsed_seconds": time.monotonic() - started,
        "reload_verified": True,
    }
    atomic_json(folder / f"{kind}_checkpoint.json", data)
    return data


def augmented_training(folder, train, labels, config, context):
    marker, path = folder / "smote_checkpoint.json", folder / "synthetic_features.npy"
    if marker.exists():
        data = read_json(marker)
        if data["context"] != context or data["sha256"] != file_sha256(path):
            raise ValueError("SMOTE checkpoint does not match training inputs")
        synthetic = np.load(path, allow_pickle=False)
        if synthetic.shape != (data["audit"]["synthetic_rows"], len(FEATURE_COLUMNS)):
            raise ValueError("Synthetic checkpoint shape mismatch")
        return np.concatenate([train, synthetic]), np.concatenate(
            [labels, np.ones(len(synthetic), dtype=labels.dtype)]
        )
    values, targets, audit = resample_training(train, labels, config)
    partial = path.with_name(path.name + ".partial")
    with partial.open("wb") as handle:
        np.save(handle, values[len(train) :], allow_pickle=False)
    partial.replace(path)
    atomic_json(
        marker, {"context": context, "sha256": file_sha256(path), "audit": audit}
    )
    return values, targets


@contextmanager
def heartbeat(log, activity):
    stopped = threading.Event()
    started = time.monotonic()

    def report():
        while not stopped.wait(45):
            log(
                f"{activity}: running for {(time.monotonic() - started) / 60:.1f} minutes"
            )

    worker = threading.Thread(target=report, daemon=True)
    worker.start()
    try:
        yield
    finally:
        stopped.set()
        worker.join()


def train_missing(output, settings, config, plan, inventory, prep, log):
    parameters = forest_parameters(config, jobs=1)
    pending, checkpoints = [], []
    for item in plan:
        if item["source"]:
            continue
        params = parameters | {
            "max_depth": item["max_depth"],
            "min_samples_leaf": item["min_samples_leaf"],
        }
        context = training_context(config, inventory, params)
        for kind in KINDS:
            checkpoint = checkpoint_model(
                output / "models" / item["candidate"], kind, context
            )
            if checkpoint is None:
                pending.append((item, kind, params, context))
            else:
                checkpoints.append(
                    {"candidate": item["candidate"], "model": kind} | checkpoint
                )
    if not pending and (output / "shap_background.parquet").exists():
        return checkpoints
    memory_preflight(
        prep["splits"]["train"]["rows"], prep["splits"]["train"]["legitimate"], 0.01
    )
    log("Loading original training split only")
    frame, labels, train_ids = load_prepared_split(Path(settings["prepared"]), "train")
    train = feature_matrix(frame)
    labels, train_ids = np.asarray(labels), np.asarray(train_ids)
    del frame
    positions = sample_positions(np.arange(len(train)), 200, 42)
    pd.DataFrame(train[positions], columns=FEATURE_COLUMNS).assign(
        source_row_number=train_ids[positions]
    ).to_parquet(output / "shap_background.parquet", index=False)
    del train_ids
    augmented = augmented_y = None
    if any(kind == "rf_smote" for _, kind, _, _ in pending):
        log("Creating or loading the shared training-only SMOTE sample")
        with heartbeat(log, "SMOTE"):
            augmented, augmented_y = augmented_training(
                output,
                train,
                labels,
                config,
                training_context(config, inventory, parameters),
            )
    for item, kind, params, context in pending:
        name = f"{item['candidate']}/{kind}"
        log(f"Training {name}; 200 trees, one worker")
        values, y = (train, labels) if kind == "rf" else (augmented, augmented_y)
        with heartbeat(log, f"Training {name}"):
            checkpoint = fit_one(
                output / "models" / item["candidate"], kind, values, y, params, context
            )
        checkpoints.append({"candidate": item["candidate"], "model": kind} | checkpoint)
        atomic_json(output / "training_progress.json", {"completed": checkpoints})
        log(
            f"Saved and reload-verified {name}; {checkpoint['elapsed_seconds'] / 60:.1f} minutes"
        )
        gc.collect()
    return checkpoints


def load_models(item, output, config, inventory):
    if item["source"]:
        return load_bundle(Path(item["source"])).models
    params = forest_parameters(config, jobs=1) | {
        "max_depth": item["max_depth"],
        "min_samples_leaf": item["min_samples_leaf"],
    }
    context = training_context(config, inventory, params)
    folder = output / "models" / item["candidate"]
    for kind in KINDS:
        if checkpoint_model(folder, kind, context) is None:
            raise ValueError("Missing completed model checkpoint")
    models = {kind: joblib.load(folder / f"{kind}.joblib") for kind in KINDS}
    for model in models.values():
        if any(model.get_params()[key] != value for key, value in params.items()):
            raise ValueError("Saved model parameters mismatch")
    return models


def evaluate_pair(
    output,
    item,
    models,
    X,
    y,
    ids,
    cohorts,
    background,
    sensitivity,
    preprocessor,
    policy,
    context,
    log,
):
    name = item["candidate"]
    folder = output / "evaluation" / name
    folder.mkdir(parents=True, exist_ok=True)
    scores = {kind: predict(models[kind], X) for kind in KINDS}
    if cohorts is None:
        cohorts = make_cohorts(
            y,
            scores,
            {
                "seed": 42,
                "shap_uniform_rows": 200,
                "shap_fraud_rows": 100,
                "shap_baseline_flagged_rows": 50,
            },
        )
    chosen = np.unique(np.concatenate(list(cohorts.values())))
    decision, grid = percent_grid_thresholds(y, scores["rf"], scores["rf_smote"])
    check = verify_grid(y, scores, decision, grid)
    atomic_json(folder / "decision.json", decision | {"verification": check})
    grid.to_csv(folder / "thresholds.csv", index=False, float_format="%.17g")
    predictions = pd.DataFrame({"source_row_number": ids, "actual_label": y})
    rows, shap_rows, sweeps, diagnostics = [], [], [], []
    for kind, model in models.items():
        predictions[f"{kind}_score"] = scores[kind]
        for mode, threshold in (
            ("fixed_43", 0.43),
            ("selected", decision["threshold"]),
        ):
            row = metric_row(y, scores[kind], threshold, policy, name, kind, mode)
            check_counts(row, y, scores[kind])
            rows.append(row)
            predictions[f"{kind}_{mode}_label"] = (scores[kind] >= threshold).astype(
                "uint8"
            )
        log(
            f"SHAP {name}/{kind}: {len(chosen)} common records; selected cutoff {decision['threshold']:.0%}"
        )
        with heartbeat(log, f"SHAP {name}/{kind}"):
            phi, base, error = shap_values(model, background, X[chosen])
        explanation = pd.DataFrame(phi, columns=FEATURE_COLUMNS)
        explanation.insert(0, "source_row_number", ids[chosen])
        explanation["base_score"], explanation["score"] = base, scores[kind][chosen]
        explanation.to_parquet(folder / f"{kind}_shap.parquet", index=False)
        for cohort, positions in cohorts.items():
            shap_rows.append(
                {"candidate": name, "model": kind, "cohort": cohort}
                | time_summary(phi[np.searchsorted(chosen, positions)])
            )
        sweep, per_row = time_sweep(
            model, X[sensitivity], ids[sensitivity], preprocessor
        )
        per_row["crosses_selected_cutoff"] = (
            per_row.minimum_score < decision["threshold"]
        ) & (per_row.maximum_score >= decision["threshold"])
        sweep.to_parquet(folder / f"{kind}_time_sweep.parquet", index=False)
        per_row.to_csv(
            folder / f"{kind}_time_sensitivity.csv", index=False, float_format="%.17g"
        )
        sweeps.append(
            {
                "candidate": name,
                "model": kind,
                "rows": len(per_row),
                "mean_range_pp": float(per_row.range_pp.mean()),
                "median_range_pp": float(per_row.range_pp.median()),
                "crosses_at_43": float(per_row.crosses_43_percent.mean()),
                "crosses_at_selected": float(per_row.crosses_selected_cutoff.mean()),
            }
        )
        np.testing.assert_allclose(
            predict(model, X[sensitivity]),
            scores[kind][sensitivity],
            rtol=0,
            atol=1e-12,
        )
        diagnostics.append(
            {
                "candidate": name,
                "model": kind,
                "shap_rows": len(chosen),
                "shap_max_error": error,
                "score_replay_rows": len(sensitivity),
                "rows": len(y),
                "time_sweep_scores": len(sweep),
            }
        )
    predictions["selected_threshold"] = decision["threshold"]
    predictions.to_parquet(folder / "predictions.parquet", index=False)
    result = {
        "rows": rows,
        "shap": shap_rows,
        "sweeps": sweeps,
        "diagnostics": diagnostics,
        "decision": decision | {"candidate": name},
    }
    atomic_json(folder / "results.json", result)
    atomic_json(
        folder / "checkpoint.json", {"context": context, "files": files_in(folder)}
    )
    return result, cohorts


def run(config_path, run_id=None, execute=False, resume=None):
    started = time.monotonic()
    config_path = Path(config_path).resolve()
    settings = read_config(config_path)
    if resume:
        if (
            not execute
            or run_id
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,100}", resume)
        ):
            raise ValueError("Use --execute --resume RUN_ID without --run-id")
        output = ROOT / "outputs" / resume
        manifest = read_json(output / "manifest.json")
        if manifest["status"] not in ("running", "failed", "interrupted"):
            raise ValueError("Only an unfinished execution can resume")
        if manifest["mode"] != "execute":
            raise ValueError("Create a separate execution after preflight")
    else:
        output = new_run(
            ROOT / "outputs", "forest_200" if execute else "preflight", run_id
        )
        manifest = {
            "schema_version": 1,
            "kind": "isolated_200_tree_regularization",
            "mode": "execute" if execute else "preflight",
            "status": "running",
            "created_at": datetime.now(UTC).isoformat(),
            "official_test_evaluated": False,
            "data_provenance_audited": False,
            "completed": [],
            "new_fits_planned": 6,
            "identity": "1-based CSV data-row positions; namespace is validation_csv_sha256",
            "elapsed_seconds": 0,
        }
    lock = output / ".run.lock"
    try:
        handle = lock.open("x", encoding="utf-8")
    except FileExistsError as exc:
        raise ValueError(
            "Run lock exists. Confirm no previous process remains before removing this run's .run.lock"
        ) from exc
    handle.write(str(os.getpid()))
    handle.close()

    def save():
        atomic_json(output / "manifest.json", manifest)

    def log(message):
        print(message, flush=True)
        with (output / "run.log").open("a", encoding="utf-8") as stream:
            stream.write(datetime.now(UTC).isoformat() + " " + message + "\n")

    try:
        log("Checking saved model references, training protocol, and CSV format")
        base_config, plan, inventory, prep = inspect_sources(settings)
        inventory[str(config_path)] = file_sha256(config_path)
        implementation = {
            str(p): file_sha256(p)
            for p in [
                *ROOT.glob("*.py"),
                *(REPO / "backend/src/integritree/ml").glob("*.py"),
                *(
                    HELPERS / name
                    for name in (
                        "experiment.py",
                        "evaluate_csv.py",
                        "compare_cutoffs.py",
                    )
                ),
            ]
        }
        packages = {p: version(p) for p in (*PACKAGES, "shap")}
        identity = {
            "settings": settings,
            "input_sha256": inventory,
            "implementation_sha256": implementation,
            "packages": packages,
            "plan": plan,
        }
        if resume and manifest.get("identity_signature") != signature(identity):
            raise ValueError(
                "Resume inputs, configuration, code, or packages differ from the original run"
            )
        config = ExperimentConfig.model_validate(base_config)
        X, y, ids = load_csv(Path(settings["validation_csv"]))
        manifest.update(
            identity,
            identity_signature=signature(identity),
            status="running",
            rows=len(y),
            actual_fraud=int(y.sum()),
            validation_csv_sha256=inventory[settings["validation_csv"]],
            baseline_configuration=base_config,
        )
        try:
            resource = memory_preflight(
                prep["splits"]["train"]["rows"],
                prep["splits"]["train"]["legitimate"],
                0.01,
            ) | {"passed": True}
        except MemoryError as exc:
            resource = {"passed": False, "error": str(exc)}
        manifest["memory_preflight"] = resource
        atomic_json(
            output / "plan.json",
            {
                "configurations": plan,
                "settings": settings,
                "baseline_configuration": base_config,
                "memory_preflight": resource,
            },
        )
        save()
        if not execute:
            manifest["status"] = "preflight_complete"
            log("Preflight complete: three reused pairs, six new fits planned")
            return output
        if not resource["passed"]:
            raise MemoryError(resource["error"])
        preprocessor = FittedPreprocessor.load(
            Path(plan[0]["source"]) / "preprocessing.json"
        )
        preprocessor.save(output / "preprocessing.json")
        checkpoints = train_missing(
            output, settings, config, plan, inventory, prep, log
        )
        manifest["trained_models"] = checkpoints
        manifest["new_fits_completed"] = len(checkpoints)
        save()
        gc.collect()
        background_file = pd.read_parquet(output / "shap_background.parquet")
        background = background_file[FEATURE_COLUMNS].to_numpy(dtype=float)
        background_hash = file_sha256(output / "shap_background.parquet")
        context = signature(
            {
                "run": manifest["identity_signature"],
                "background": background_hash,
                "models": checkpoints,
            }
        )
        sensitivity = sample_positions(np.arange(len(y)), 50, 42)
        pd.DataFrame({"source_row_number": ids[sensitivity]}).to_csv(
            output / "sensitivity_cohort.csv", index=False
        )
        cohorts, results = None, []
        for item in plan:
            name = item["candidate"]
            folder = output / "evaluation" / name
            marker = folder / "checkpoint.json"
            if marker.exists():
                checkpoint = read_json(marker)
                if checkpoint["context"] != context:
                    raise ValueError("Evaluation checkpoint context mismatch")
                validate_files(folder, checkpoint["files"])
                result = read_json(folder / "results.json")
                if cohorts is None:
                    frame = pd.read_parquet(folder / "predictions.parquet")
                    np.testing.assert_array_equal(frame.actual_label, y)
                    np.testing.assert_array_equal(frame.source_row_number, ids)
                    cohorts = make_cohorts(
                        y,
                        {kind: frame[f"{kind}_score"].to_numpy() for kind in KINDS},
                        settings,
                    )
                log(f"Reusing completed evaluation {name}")
            else:
                log(f"Scoring and explaining {name} on supplied validation CSV")
                models = load_models(item, output, config, inventory)
                result, cohorts = evaluate_pair(
                    output,
                    item,
                    models,
                    X,
                    y,
                    ids,
                    cohorts,
                    background,
                    sensitivity,
                    preprocessor,
                    config.evaluation,
                    context,
                    log,
                )
                del models
                gc.collect()
            results.append(result)
            pd.DataFrame(
                [
                    {
                        "cohort": name,
                        "source_row_number": int(ids[i]),
                        "actual_label": int(y[i]),
                    }
                    for name, positions in cohorts.items()
                    for i in positions
                ]
            ).to_csv(output / "shap_cohorts.csv", index=False)
            manifest["completed"] = [item["decision"]["candidate"] for item in results]
            save()
        from reporting import report

        report(output, results, manifest)
        from verify import verify_results

        verification = verify_results(output, manifest)
        verify_unchanged(inventory)
        verify_unchanged(implementation)
        atomic_json(
            output / "verification.json",
            verification | {"source_preservation_verified": True},
        )
        manifest.update(
            status="complete",
            source_preservation_verified=True,
            completed_at=datetime.now(UTC).isoformat(),
        )
        log(
            "Completed all six pairs; verification passed; no application settings changed"
        )
        return output
    except BaseException as exc:
        manifest.update(
            status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            failure_type=type(exc).__name__,
            error=str(exc),
        )
        raise
    finally:
        manifest["elapsed_seconds"] += time.monotonic() - started
        if manifest["status"] == "complete":
            manifest["files"] = {
                p.relative_to(output).as_posix(): file_sha256(p)
                for p in output.rglob("*")
                if p.is_file()
                and p.name not in ("manifest.json", ".run.lock")
                and not p.name.endswith(".partial")
            }
        save()
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--run-id")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume")
    args = parser.parse_args()
    print(run(args.config, args.run_id, args.execute, args.resume))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
