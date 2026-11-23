"""Validation-only depth/leaf experiment with matched time-dependence diagnostics."""

import argparse
import gc
import json
import os
import sys
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier

from integritree.ml.artifacts import PACKAGES, load_bundle
from integritree.ml.data import file_sha256, write_json
from integritree.ml.evaluation import metrics
from integritree.ml.features import FEATURE_COLUMNS
from integritree.ml.inference import feature_matrix, fraud_scores
from integritree.ml.preparation import load_prepared_split
from integritree.ml.research import new_run, verify_prepared
from integritree.ml.shap_adapter import forest_representation
from integritree.ml.training import (
    forest_parameters,
    memory_preflight,
    resample_training,
)

KINDS = ("rf", "rf_smote")
TIME = [FEATURE_COLUMNS.index(n) for n in ("hour_of_day", "day_of_week")]
GROUPS = {
    "time": TIME,
    "type": [i for i, n in enumerate(FEATURE_COLUMNS) if n.startswith("type_")],
    "amount": [FEATURE_COLUMNS.index(n) for n in ("log_amount", "is_zero_amount")],
    "entity": [
        FEATURE_COLUMNS.index(n) for n in ("is_merchant_origin", "is_merchant_dest")
    ],
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_config(path):
    path = Path(path).resolve()
    settings = yaml.safe_load(path.read_text(encoding="utf-8"))
    keys = {
        "schema_version",
        "models",
        "prepared",
        "forests",
        "reuse_existing",
        "seed",
        "shap_uniform_rows",
        "shap_fraud_rows",
        "shap_baseline_flagged_rows",
        "background_rows",
        "sensitivity_rows",
    }
    if (
        not isinstance(settings, dict)
        or set(settings) != keys
        or settings["schema_version"] != 1
    ):
        raise ValueError("Unsupported experiment configuration")
    for name in keys - {
        "schema_version",
        "models",
        "prepared",
        "forests",
        "reuse_existing",
    }:
        if type(settings[name]) is not int or settings[name] < 1:
            raise ValueError(f"Invalid {name}")
    if settings["seed"] != 42 or type(settings["reuse_existing"]) is not bool:
        raise ValueError("Use seed 42 and a boolean reuse_existing")
    if settings["background_rows"] != 200:
        raise ValueError("Keep the reference background size of 200")
    grid = settings["forests"]
    if not isinstance(grid, list) or not 2 <= len(grid) <= 12 or grid[0] != [10, 1]:
        raise ValueError("Start with baseline [10, 1]; supply 2 to 12 forest pairs")
    for pair in grid:
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or any(type(v) is not int for v in pair)
            or not 1 <= pair[0] <= 10
            or pair[1] < 1
        ):
            raise ValueError(
                "Only reduced depth (1..10) / increased leaf size (>=1) are supported"
            )
    if len(set(map(tuple, grid))) != len(grid):
        raise ValueError("Duplicate forest configurations")
    for name in ("models", "prepared"):
        if not isinstance(settings[name], str) or not settings[name].strip():
            raise ValueError(f"Missing {name} path")
        settings[name] = str((path.parent / settings[name]).resolve())
    return settings


def key(depth, leaf):
    return f"depth_{depth}_leaf_{leaf}"


def predict(model, values):
    return np.concatenate(
        [
            fraud_scores(model, values[i : i + 25000])
            for i in range(0, len(values), 25000)
        ]
    )


def verify_reference(bundle):
    c = bundle.config
    expected = forest_parameters(c, jobs=1)
    if (
        expected["n_estimators"] != 100
        or expected["max_depth"] != 10
        or expected["min_samples_leaf"] != 1
        or c.smote.sampling_ratio != 0.01
        or c.smote.method != "smote_encoded"
        or c.smote.k_neighbors != 5
        or c.scoring.threshold != 0.43
        or c.seeds.model != 42
        or c.seeds.smote != 42
    ):
        raise ValueError(
            "Reference must be the selected 100-tree/depth-10/leaf-1, 1:100, 43% pair"
        )
    for model in bundle.models.values():
        if any(model.get_params()[n] != v for n, v in expected.items()):
            raise ValueError("Reference model/configuration mismatch")
    return expected


def verify_reuse(candidate, reference, parameters):
    for name in ("dataset", "preprocessing", "split", "seeds", "smote"):
        if getattr(candidate.config, name) != getattr(reference.config, name):
            raise ValueError(f"Reusable candidate differs in {name}")
    if (
        candidate.metadata["prepared_metadata_sha256"]
        != reference.metadata["prepared_metadata_sha256"]
    ):
        raise ValueError("Reusable candidate has different prepared data")
    if candidate.preprocessor.state != reference.preprocessor.state:
        raise ValueError("Reusable candidate has different fitted preprocessing")
    for model in candidate.models.values():
        expected = reference.models["rf"].get_params() | parameters
        if model.get_params() != expected:
            raise ValueError("Reusable candidate differs in forest parameters")


def bundle_inventory(path):
    """Fingerprint declared files, including a selected bundle's referenced model."""
    path = path.resolve()
    metadata = read_json(path / "metadata.json")
    files = [path / "metadata.json"]
    for name in metadata["files"]:
        target = (path / name).resolve()
        if not target.is_relative_to(path):
            raise ValueError("Bundle path traversal")
        files.append(target)
    result = {str(p): file_sha256(p) for p in files}
    if metadata.get("kind") == "validation_selection":
        from integritree.ml.staged_selection import resolve_reference

        selection = read_json(path / "selection.json")
        result.update(bundle_inventory(resolve_reference(selection["model"], path)))
    return result


def prepare_plan(settings):
    models, prepared = Path(settings["models"]), Path(settings["prepared"])
    reference = load_bundle(models)
    parameters = verify_reference(reference)
    prep = verify_prepared(reference, prepared)
    inventory = bundle_inventory(models)
    for name in (
        "metadata.json",
        "configuration.json",
        "preprocessing.json",
        "split_manifest.parquet",
        "train_features.parquet",
        "validation_features.parquet",
    ):
        p = prepared / name
        actual = file_sha256(p)
        if name != "metadata.json" and actual != prep["files"][name]:
            raise ValueError(f"Prepared fingerprint mismatch: {name}")
        inventory[str(p)] = actual
    plan = []
    for depth, leaf in settings["forests"]:
        item = {
            "candidate": key(depth, leaf),
            "max_depth": depth,
            "min_samples_leaf": leaf,
            "source": None,
            "new_fits": 2,
        }
        if (depth, leaf) == (10, 1):
            item.update(source=str(models), new_fits=0)
        elif settings["reuse_existing"]:
            parent = (
                models
                / "candidates"
                / f"ratio_0.01_trees_100_depth_{depth}_leaf_{leaf}"
            )
            options = sorted(
                p.parent
                for p in parent.glob("*/metadata.json")
                if read_json(p).get("status") == "complete"
            )
            if len(options) > 1:
                raise ValueError(f"Ambiguous reusable candidates: {parent}")
            if options:
                candidate = load_bundle(options[0])
                verify_reuse(
                    candidate,
                    reference,
                    parameters | {"max_depth": depth, "min_samples_leaf": leaf},
                )
                inventory.update(bundle_inventory(options[0]))
                item.update(source=str(options[0]), new_fits=0)
                del candidate
        plan.append(item)
    try:
        resource = memory_preflight(
            prep["splits"]["train"]["rows"], prep["splits"]["train"]["legitimate"], 0.01
        )
        resource["passed"] = True
    except MemoryError as exc:
        resource = {"passed": False, "error": str(exc)}
    return reference, plan, inventory, resource


def verify_unchanged(inventory):
    for name, expected in inventory.items():
        if file_sha256(Path(name)) != expected:
            raise ValueError(f"Source changed during experiment: {name}")


def sample_positions(pool, count, seed):
    pool = np.asarray(pool)
    return np.sort(
        np.random.default_rng(seed).choice(pool, min(count, len(pool)), replace=False)
    )


def make_cohorts(labels, baseline_scores, settings):
    # Freeze membership before inspecting any candidate. Cohorts deliberately overlap.
    cohorts = {
        "uniform_validation": sample_positions(
            np.arange(len(labels)), settings["shap_uniform_rows"], settings["seed"]
        ),
        "actual_fraud": sample_positions(
            np.flatnonzero(labels == 1), settings["shap_fraud_rows"], settings["seed"]
        ),
    }
    for name in KINDS:
        cohorts[f"baseline_flagged_{name}"] = sample_positions(
            np.flatnonzero(baseline_scores[name] >= 0.43),
            settings["shap_baseline_flagged_rows"],
            settings["seed"],
        )
    return cohorts


def shap_values(model, background, values):
    import shap

    masker = shap.maskers.Independent(background, max_samples=len(background))
    explainer = shap.TreeExplainer(
        forest_representation(model),
        masker,
        model_output="probability",
        feature_perturbation="interventional",
    )
    if len(explainer.data) != len(background):
        raise ValueError("SHAP background was subsampled unexpectedly")
    np.testing.assert_allclose(
        explainer.model.predict(background),
        model.predict_proba(background),
        atol=1e-12,
        rtol=0,
    )
    result = explainer(values.astype(np.float32), check_additivity=True)
    if result.values.shape != (len(values), len(FEATURE_COLUMNS), 2):
        raise ValueError("Unexpected SHAP shape")
    phi, base = result.values[:, :, 1], result.base_values[:, 1]
    error = np.max(np.abs(base + phi.sum(axis=1) - predict(model, values)))
    if error > 1e-6:
        raise ValueError(f"SHAP score reconstruction failed: {error}")
    return phi, base, float(error)


def time_summary(phi):
    """Absolute values avoid cancellation; pp means score percentage points."""
    if not len(phi):
        return {
            "rows": 0,
            "mean_time_abs_pp": None,
            "mean_time_net_abs_pp": None,
            "time_share_abs": None,
            "time_strictly_largest_group_fraction": None,
        }
    absolute = np.abs(phi)
    time = absolute[:, TIME].sum(axis=1)
    total = float(absolute.sum())
    groups = np.column_stack(
        [absolute[:, indices].sum(axis=1) for indices in GROUPS.values()]
    )
    return {
        "rows": len(phi),
        "mean_time_abs_pp": float(time.mean() * 100),
        "mean_time_net_abs_pp": float(np.abs(phi[:, TIME].sum(axis=1)).mean() * 100),
        "time_share_abs": float(time.sum() / total) if total else None,
        "time_strictly_largest_group_fraction": float(
            (groups[:, 0] > groups[:, 1:].max(axis=1)).mean()
        ),
    }


def time_sweep(model, values, ids, preprocessor):
    """Diagnostic substitutions, not labeled or necessarily in-distribution records."""
    hours, days = np.tile(np.arange(24), 7), np.repeat(np.arange(7), 24)
    grid = np.repeat(values, 168, axis=0)
    for name, raw in (("hour_of_day", hours), ("day_of_week", days)):
        j = preprocessor.state.scaled_columns.index(name)
        grid[:, FEATURE_COLUMNS.index(name)] = (
            np.tile(raw, len(values)) * preprocessor.state.scale[j]
            + preprocessor.state.offset[j]
        )
    scores = predict(model, grid).reshape(len(values), 168)
    details = pd.DataFrame(
        {
            "source_row_number": np.repeat(ids, 168),
            "simulation_hour": np.tile(hours, len(values)),
            "simulation_day": np.tile(days, len(values)),
            "score": scores.ravel(),
        }
    )
    low, high = scores.min(axis=1), scores.max(axis=1)
    per_row = pd.DataFrame(
        {
            "source_row_number": ids,
            "minimum_score": low,
            "maximum_score": high,
            "range_pp": 100 * (high - low),
            "crosses_43_percent": (low < 0.43) & (high >= 0.43),
        }
    )
    return details, per_row


def metric_row(labels, scores, policy, **fields):
    result = metrics(labels, scores, 0.43, policy)
    row = fields | {n: m["value"] for n, m in result["metrics"].items()}
    row.update(
        {n + "_undefined_reason": m["reason"] for n, m in result["metrics"].items()}
    )
    row.update({n: result["confusion_matrix"][n] for n in ("tp", "fp", "tn", "fn")})
    row.update(rows=len(labels), actual_fraud=int(np.sum(labels)), threshold=0.43)
    row["false_alarms_per_10000_legitimate"] = (
        10000 * row["fp"] / (row["fp"] + row["tn"]) if row["fp"] + row["tn"] else None
    )
    return row


def save_report(output, rows, shap_rows, sensitivity_rows):
    import html

    tables = {
        "metrics": pd.DataFrame(rows),
        "time_shap": pd.DataFrame(shap_rows),
        "time_sensitivity": pd.DataFrame(sensitivity_rows),
    }
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False, float_format="%.17g")
    baseline = (
        tables["metrics"].query("candidate == 'depth_10_leaf_1'").set_index("model")
    )
    changes = tables["metrics"].copy()
    for name in ("precision", "recall", "f1", "mcc", "pr_auc", "tp", "fp"):
        changes[f"delta_{name}"] = [
            r[name] - baseline.loc[r["model"], name]
            if r[name] is not None and pd.notna(baseline.loc[r["model"], name])
            else None
            for r in rows
        ]
    changes.to_csv(
        output / "changes_vs_baseline.csv", index=False, float_format="%.17g"
    )
    note = (
        "Exploratory validation results. All models use the same 43% cutoff and 1:100 SMOTE protocol. "
        "SHAP cohorts are fixed and overlapping; only the uniform cohort estimates the validation distribution. "
        "Time substitutions are sensitivity diagnostics, not real transactions or accuracy evidence. "
        "No candidate is automatically selected or deployed. Test data were not evaluated. "
        "These tables may be a partial checkpoint: confirm status=complete in manifest.json."
    )
    body = [
        "<!doctype html><meta charset='utf-8'><title>Forest regularization experiment</title>",
        (
            "<style>body{font:16px system-ui;margin:2rem;color:#182333}table{border-collapse:collapse;font-size:13px}"
            "td,th{padding:7px;border:1px solid #ccc;text-align:right}section{overflow-x:auto;margin-bottom:2rem}</style>"
        ),
        "<h1>Depth and minimum leaf size</h1><p>" + html.escape(note) + "</p>",
    ]
    for name, table in tables.items():
        body += [
            "<h2>" + html.escape(name.replace("_", " ")) + "</h2><section>",
            table.to_html(index=False, float_format=lambda v: f"{v:.6g}", escape=True),
            "</section>",
        ]
    (output / "report.html").write_text("\n".join(body), encoding="utf-8")
    (output / "SUMMARY.md").write_text(
        "# Forest regularization experiment\n\n" + note + "\n\n"
        "Open report.html for tables. metrics.csv and changes_vs_baseline.csv use full validation. "
        "time_shap.csv and time_sensitivity.csv describe the fixed samples. Lower time influence "
        "alone is not a success criterion: inspect lost fraud detections and false alarms together.\n",
        encoding="utf-8",
    )


def run(root, config_path, run_id=None, execute=False):
    runtime = root / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    os.environ["NUMBA_CACHE_DIR"] = str(runtime / "numba")
    os.environ["MPLCONFIGDIR"] = str(runtime / "matplotlib")
    settings = read_config(config_path)
    output = new_run(
        root / "outputs", "forest_regularization" if execute else "preflight", run_id
    )
    manifest = {
        "schema_version": 1,
        "kind": "isolated_forest_regularization",
        "status": "running",
        "mode": "execute" if execute else "preflight",
        "created_at": datetime.now(UTC).isoformat(),
        "test_used": False,
        "settings": settings,
        "fits_started": 0,
        "completed": [],
        "packages": {p: version(p) for p in (*PACKAGES, "shap", "PyYAML")},
    }

    def save():
        write_json(output / "manifest.json", manifest)

    def progress(message):
        print(message, file=sys.stderr, flush=True)
        with (output / "run.log").open("a", encoding="utf-8") as handle:
            handle.write(datetime.now(UTC).isoformat() + " " + message + "\n")

    save()
    try:
        progress("Verifying reference, prepared fingerprints, and reusable candidates")
        reference, plan, inventory, resource = prepare_plan(settings)
        inventory[str(Path(config_path).resolve())] = file_sha256(Path(config_path))
        code = list(root.glob("*.py")) + list(
            (root.parent / "src" / "integritree").rglob("*.py")
        )
        manifest.update(
            plan=plan,
            resource_preflight=resource,
            input_sha256=inventory,
            implementation_sha256={str(p): file_sha256(p) for p in code},
            new_fits_planned=sum(p["new_fits"] for p in plan),
        )
        write_json(
            output / "plan.json",
            {
                "candidates": plan,
                "resource_preflight": resource,
                "fixed_configuration": reference.config.model_dump(),
            },
        )
        save()
        progress(
            f"Plan verified: {len(plan)} paired configurations, {manifest['new_fits_planned']} new fits"
        )
        if not execute:
            manifest["status"] = "preflight_complete"
            save()
            return output
        if not resource["passed"]:
            raise MemoryError(resource["error"])
        prepared = Path(settings["prepared"])
        reference.preprocessor.save(output / "preprocessing.json")
        progress(
            "Loading original train and validation only; checking disjoint identities"
        )
        train_frame, labels, train_ids = load_prepared_split(prepared, "train")
        val_frame, truth, val_ids = load_prepared_split(prepared, "validation")
        train, validation = feature_matrix(train_frame), feature_matrix(val_frame)
        del train_frame, val_frame
        labels, truth, train_ids, val_ids = map(
            np.asarray, (labels, truth, train_ids, val_ids)
        )
        if (
            np.intersect1d(train_ids, val_ids).size
            or set(np.unique(labels)) != {0, 1}
            or set(np.unique(truth)) != {0, 1}
        ):
            raise ValueError("Overlapping partitions or missing label classes")
        baseline_scores = {
            name: predict(reference.models[name], validation) for name in KINDS
        }
        cohorts = make_cohorts(truth, baseline_scores, settings)
        chosen = np.unique(np.concatenate(list(cohorts.values())))
        background_pos = sample_positions(
            np.arange(len(train)), settings["background_rows"], settings["seed"]
        )
        background = train[background_pos].copy()
        pd.DataFrame(background, columns=FEATURE_COLUMNS).assign(
            source_row_number=train_ids[background_pos]
        ).to_parquet(output / "shap_background.parquet", index=False)
        cohort_rows = [
            {
                "cohort": name,
                "source_row_number": int(val_ids[i]),
                "actual_label": int(truth[i]),
            }
            for name, positions in cohorts.items()
            for i in positions
        ]
        pd.DataFrame(cohort_rows).to_csv(output / "shap_cohorts.csv", index=False)
        sensitivity = sample_positions(
            np.arange(len(truth)), settings["sensitivity_rows"], settings["seed"]
        )
        pd.DataFrame({"source_row_number": val_ids[sensitivity]}).to_csv(
            output / "sensitivity_cohort.csv", index=False
        )
        # Exactly one training-only resample is shared by every newly fitted SMOTE candidate.
        if any(p["new_fits"] for p in plan):
            progress(
                "Creating one shared ordinary-SMOTE training matrix at ratio 1:100"
            )
            augmented, augmented_y, audit = resample_training(
                train, labels, reference.config
            )
            write_json(output / "smote_audit.json", audit)
            pd.DataFrame(augmented[len(train) :], columns=FEATURE_COLUMNS).to_parquet(
                output / "synthetic_features.parquet", index=False
            )
        else:
            augmented = augmented_y = None
        rows, summaries, sweeps = [], [], []
        parameters = forest_parameters(reference.config, jobs=1)
        for item in plan:
            name = item["candidate"]
            destination = output / name
            destination.mkdir()
            progress(
                f"Evaluating {name}: {'verified saved pair' if item['source'] else 'new paired fits'}"
            )
            candidate = (
                reference
                if name == key(10, 1)
                else load_bundle(Path(item["source"]))
                if item["source"]
                else None
            )
            candidate_parameters = parameters | {
                "max_depth": item["max_depth"],
                "min_samples_leaf": item["min_samples_leaf"],
            }
            if candidate is not None and candidate is not reference:
                verify_reuse(candidate, reference, candidate_parameters)
            predictions = pd.DataFrame(
                {"source_row_number": val_ids, "actual_label": truth}
            )
            diagnostics = {}
            for kind in KINDS:
                if candidate is None:
                    manifest["fits_started"] += 1
                    save()
                    progress(f"Training {name}/{kind} with one worker")
                    X, y = (train, labels) if kind == "rf" else (augmented, augmented_y)
                    model = RandomForestClassifier(**candidate_parameters).fit(X, y)
                    joblib.dump(model, destination / f"{kind}.joblib", compress=3)
                    expected = predict(model, validation[:128])
                    del model
                    model = joblib.load(destination / f"{kind}.joblib")
                    np.testing.assert_allclose(
                        predict(model, validation[:128]), expected, atol=1e-12, rtol=0
                    )
                else:
                    model = candidate.models[kind]
                scores = predict(model, validation)
                predictions[f"{kind}_score"] = scores
                predictions[f"{kind}_predicted_label"] = (scores >= 0.43).astype(
                    "uint8"
                )
                row = metric_row(
                    truth,
                    scores,
                    reference.config.evaluation,
                    candidate=name,
                    model=kind,
                    max_depth=item["max_depth"],
                    min_samples_leaf=item["min_samples_leaf"],
                )
                rows.append(row)
                progress(
                    f"Explaining {name}/{kind} on {len(chosen)} fixed validation records"
                )
                phi, base, error = shap_values(model, background, validation[chosen])
                explanation = pd.DataFrame(phi, columns=FEATURE_COLUMNS)
                explanation.insert(0, "source_row_number", val_ids[chosen])
                explanation["base_score"] = base
                explanation["score"] = scores[chosen]
                explanation.to_parquet(
                    destination / f"{kind}_shap.parquet", index=False
                )
                for cohort, positions in cohorts.items():
                    subset = phi[np.searchsorted(chosen, positions)]
                    summaries.append(
                        {"candidate": name, "model": kind, "cohort": cohort}
                        | time_summary(subset)
                    )
                grid, per_row = time_sweep(
                    model,
                    validation[sensitivity],
                    val_ids[sensitivity],
                    reference.preprocessor,
                )
                grid.to_parquet(destination / f"{kind}_time_sweep.parquet", index=False)
                per_row.to_csv(
                    destination / f"{kind}_time_sensitivity.csv", index=False
                )
                sweeps.append(
                    {
                        "candidate": name,
                        "model": kind,
                        "rows": len(per_row),
                        "mean_range_pp": float(per_row.range_pp.mean()),
                        "median_range_pp": float(per_row.range_pp.median()),
                        "crosses_cutoff_fraction": float(
                            per_row.crosses_43_percent.mean()
                        ),
                    }
                )
                diagnostics[kind] = {
                    "shap_additivity_max_error": error,
                    "mean_actual_tree_depth": float(
                        np.mean([t.tree_.max_depth for t in model.estimators_])
                    ),
                    "total_leaves": int(
                        sum(t.tree_.n_leaves for t in model.estimators_)
                    ),
                }
                del model
                gc.collect()
            predictions.to_parquet(destination / "predictions.parquet", index=False)
            write_json(
                destination / "metadata.json",
                {
                    "kind": "isolated_forest_regularization_candidate",
                    "status": "complete",
                    "protocol": item,
                    "parameters": candidate_parameters,
                    "threshold": 0.43,
                    "prepared_metadata_sha256": reference.metadata[
                        "prepared_metadata_sha256"
                    ],
                    "diagnostics": diagnostics,
                    "test_used": False,
                },
            )
            manifest["completed"].append(name)
            save_report(output, rows, summaries, sweeps)
            save()
            del candidate
            gc.collect()
        progress("Checking source preservation and writing final output fingerprints")
        verify_unchanged(inventory)
        verify_unchanged(manifest["implementation_sha256"])
        manifest.update(
            status="complete",
            source_preservation_verified=True,
            files={
                p.relative_to(output).as_posix(): file_sha256(p)
                for p in output.rglob("*")
                if p.is_file() and p.name != "manifest.json"
            },
        )
        save()
        return output
    except BaseException as exc:
        manifest.update(
            status="failed", failure_type=type(exc).__name__, error=str(exc)
        )
        save()
        raise


def main(root):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "config.yaml")
    parser.add_argument("--run-id")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Train/evaluate; otherwise only verify and write a plan",
    )
    args = parser.parse_args()
    try:
        output = run(root, args.config, args.run_id, args.execute)
    except (ValueError, OSError, MemoryError, AssertionError) as exc:
        print(json.dumps({"success": False, "error": str(exc)}), file=sys.stderr)
        return 2
    print(
        json.dumps(
            {"success": True, "output": str(output), "executed": args.execute}, indent=2
        )
    )
    return 0
