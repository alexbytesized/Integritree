"""Versioned sequential ratio/AP, forest/AP, and shared-threshold/F1 selection."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from fractions import Fraction
from hashlib import sha256
from importlib.metadata import version
from itertools import product
import json
import os
from pathlib import Path
import sys
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, model_validator
import yaml

from integritree.config import ExperimentConfig, _UniqueKeyLoader, load_experiment
from integritree.ml.artifacts import PACKAGES, REQUIRED_FILES, ModelBundle, load_bundle
from integritree.ml.data import file_sha256, write_json
from integritree.ml.evaluation import metrics, validate_policy
from integritree.ml.features import FEATURE_COLUMNS
from integritree.ml.research import evaluate_run, load_report, new_run
from integritree.ml.selection import exact_mean_f1, release_candidate_memory
from integritree.ml.threshold_search import all_score_thresholds
from integritree.ml.training import forest_parameters, train_models, validate_training_config
from integritree.settings import load_settings

STAGES = ("01_smote_ratio", "02_random_forest", "03_threshold")


class SelectionProtocol(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1] = 1
    name: Literal["sequential_ratio_forest_threshold"] = "sequential_ratio_forest_threshold"
    revision: Literal[1, 2] = 1
    amendment_note: str | None = None
    ratios: list[float] = [.1, .2, 1 / 3, .5, 1.]
    reference_forest: tuple[int, int, int] = (100, 10, 1)
    trees: list[int] = [100, 200]
    depths: list[int] = [10, 20]
    leaves: list[int] = [1, 10, 50]
    ratio_objective: Literal["rf_smote_average_precision"] = "rf_smote_average_precision"
    forest_objective: Literal["mean_average_precision"] = "mean_average_precision"
    threshold_objective: Literal["mean_f1"] = "mean_f1"
    ratio_ties: Literal["smaller_ratio"] = "smaller_ratio"
    forest_ties: Literal["shallower_fewer_trees_larger_leaf"] = "shallower_fewer_trees_larger_leaf"
    threshold_candidates: Literal["distinct_scores_plus_0_half_1"] = "distinct_scores_plus_0_half_1"
    threshold_ties: Literal["closest_to_half_then_higher"] = "closest_to_half_then_higher"
    selection_split: Literal["validation"] = "validation"

    @model_validator(mode="after")
    def valid_grid(self):
        if self.revision == 2 and not (self.amendment_note and self.amendment_note.strip()):
            raise ValueError("Protocol revision 2 requires an amendment disclosure")
        if not self.ratios or any(not np.isfinite(r) or not 0 < r <= 1 for r in self.ratios):
            raise ValueError("Ratios must be finite minority/majority values in (0, 1]")
        for values in (self.ratios, self.trees, self.depths, self.leaves):
            if not values or len(set(values)) != len(values):
                raise ValueError("Candidate lists must be nonempty and unique")
        if any(v < 1 for values in (self.trees, self.depths, self.leaves) for v in values):
            raise ValueError("Forest candidates must be positive")
        if self.reference_forest not in self.forests():
            raise ValueError("Reference forest must occur in the forest search")
        return self

    def forests(self):
        return list(product(self.trees, self.depths, self.leaves))


def load_protocol(path):
    return SelectionProtocol.model_validate(yaml.load(Path(path).read_text(encoding="utf-8-sig"), Loader=_UniqueKeyLoader))


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    write_json(temporary, value)
    temporary.replace(path)


@contextmanager
def run_lock(path):
    """OS lock survives no process exit; the harmless lock file may remain."""
    with (path / ".run.lock").open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if sys.platform == "win32":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ValueError("Another worker owns this selection run") from exc
        try:
            yield
        finally:
            handle.seek(0)
            if sys.platform == "win32":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def candidate_config(base, ratio, forest):
    config = base.model_copy(deep=True)
    config.smote.sampling_ratio = ratio
    config.random_forest.n_estimators, config.random_forest.max_depth, config.random_forest.min_samples_leaf = forest
    return config


def training_identity(config, exclude_search=False, exclude_ratio=False):
    data = {name: getattr(config, name).model_dump() for name in
            ("dataset", "preprocessing", "split", "seeds", "random_forest", "smote", "scoring")}
    if exclude_search:
        for name in ("n_estimators", "max_depth", "min_samples_leaf"):
            data["random_forest"].pop(name)
    if exclude_search or exclude_ratio:
        data["smote"].pop("sampling_ratio")
    return data


def inspect_model(path, prepared_hash):
    """Verify immutable inputs without keeping large forests in memory."""
    meta = read_json(path / "metadata.json")
    if (meta.get("schema_version") != 1 or meta.get("status") != "complete"
            or meta.get("stage") != "baseline_not_final"
            or meta.get("prepared_metadata_sha256") != prepared_hash
            or meta.get("test_used") is not False or meta.get("validation_used") is not False
            or meta.get("feature_order") != FEATURE_COLUMNS
            or meta.get("rf_parameters", {}).get("n_jobs") != 1
            or meta.get("files", {}).get("prepared_metadata.json") != prepared_hash
            or set(meta.get("files", {})) != REQUIRED_FILES):
        raise ValueError(f"Incompatible candidate model provenance: {path}")
    for name, digest in meta["files"].items():
        if file_sha256(path / name) != digest:
            raise ValueError(f"Candidate model fingerprint mismatch: {path / name}")
    if any(meta.get("package_versions", {}).get(p) != version(p) for p in PACKAGES):
        raise ValueError("Candidate dependency versions differ")
    config = ExperimentConfig.model_validate(read_json(path / "configuration.json"))
    validate_training_config(config)
    if config.dataset.model_dump() != meta["dataset"]:
        raise ValueError("Candidate dataset configuration differs")
    if meta["rf_parameters"] != forest_parameters(config, meta["rf_parameters"]["n_jobs"]):
        raise ValueError("Candidate forest parameters differ")
    return meta, config


def reference(path, root):
    return {"path": os.path.relpath(path.resolve(), root.resolve()),
            "metadata_sha256": file_sha256(path / "metadata.json")}


def resolve_reference(ref, root):
    path = (root / ref["path"]).resolve()
    if file_sha256(path / "metadata.json") != ref["metadata_sha256"]:
        raise ValueError(f"Referenced metadata fingerprint mismatch: {path}")
    return path


def inventory(roots, output, base, prepared_hash, protocol):
    models, reports = [], []
    seen = set()
    for root in roots:
        for file in sorted(Path(root).rglob("metadata.json")):
            path = file.parent.resolve()
            if path in seen or path.is_relative_to(output.resolve()):
                continue
            seen.add(path)
            meta = read_json(file)
            if meta.get("status") != "complete" or meta.get("prepared_metadata_sha256") != prepared_hash:
                continue
            if meta.get("stage") == "baseline_not_final":
                config = ExperimentConfig.model_validate(read_json(path / "configuration.json"))
                if training_identity(config, True) != training_identity(base, True):
                    continue
                forest = (config.random_forest.n_estimators, config.random_forest.max_depth,
                          config.random_forest.min_samples_leaf)
                if forest not in protocol.forests():
                    continue
                inspect_model(path, prepared_hash)
                models.append(reference(path, output))
            elif meta.get("split") == "validation" and meta.get("official_test") is False:
                reports.append(reference(path, output))
    return {"models": models, "reports": reports}


def verify_report(path, model_path, config, prepared_hash):
    meta, predictions = load_report(path)
    if (meta.get("split") != "validation" or meta.get("official_test") is not False
            or meta.get("prepared_metadata_sha256") != prepared_hash
            or meta.get("dataset_sha256") != config.dataset.sha256
            or meta.get("model_metadata_sha256") != file_sha256(model_path / "metadata.json")
            or meta.get("policy") != config.evaluation.model_dump()
            or any(meta.get("package_versions", {}).get(p) != version(p)
                   for p in ("scikit-learn", "numpy", "scipy", "matplotlib"))
            or not predictions["split"].eq("validation").all()):
        raise ValueError("Candidate validation report provenance differs")
    if set(predictions.actual_label.unique()) != {0, 1}:
        raise ValueError("Candidate validation requires both classes")
    return predictions


def summarize_candidate(predictions, config):
    row = {"ratio": config.smote.sampling_ratio,
           "forest": [config.random_forest.n_estimators, config.random_forest.max_depth,
                      config.random_forest.min_samples_leaf], "reference_threshold": .5}
    for name in ("rf", "rf_smote"):
        result = metrics(predictions.actual_label, predictions[f"{name}_risk_score"], .5, config.evaluation)
        row[name] = {key: value["value"] for key, value in result["metrics"].items()}
        row[name].update({key: result["confusion_matrix"][key] for key in ("tp", "fp", "tn", "fn")})
        row[name]["score_sha256"] = sha256(np.asarray(predictions[f"{name}_risk_score"], dtype="<f8").tobytes()).hexdigest()
    row["mean_ap"] = (row["rf"]["pr_auc"] + row["rf_smote"]["pr_auc"]) / 2
    return row


def choose_ratio(rows):
    return max(rows, key=lambda r: (r["rf_smote"]["pr_auc"], -r["ratio"]))


def choose_forest(rows):
    return max(rows, key=lambda r: (r["mean_ap"], -r["forest"][1], -r["forest"][0], r["forest"][2]))


def freeze_stage(path, rows, winner, objective):
    path.mkdir(parents=True, exist_ok=True)
    decision = {"frozen": True, "objective": objective, "candidates": rows, "chosen_key": winner["key"]}
    target = path / "decision.json"
    if target.exists() and read_json(target) != decision:
        raise ValueError("Frozen stage differs from recomputed selection")
    atomic_json(target, decision)
    flat = []
    for row in rows:
        flat.append({"key": row["key"], "ratio": row["ratio"], "trees": row["forest"][0],
                     "depth": row["forest"][1], "min_samples_leaf": row["forest"][2],
                     "selected": row["key"] == winner["key"], "mean_ap": row["mean_ap"],
                     **{f"{model}_{metric}": value for model in ("rf", "rf_smote")
                        for metric, value in row[model].items()}})
    pd.DataFrame(flat).to_csv(path / "candidates.csv", index=False, float_format="%.17g")


def candidate_key(ratio, forest):
    return f"ratio_{format(ratio, '.17g')}_trees_{forest[0]}_depth_{forest[1]}_leaf_{forest[2]}"


def select_three_stage(prepared, config, protocol, output_root, run_id=None, jobs=1,
                       resume=None, reuse_roots=(), progress=print, stop_after_stage=3):
    if stop_after_stage not in (1, 3):
        raise ValueError("stop_after_stage must be 1 or 3")
    config.require_ready("evaluate")
    validate_training_config(config)
    validate_policy(config.evaluation)
    if jobs != 1:
        raise ValueError("The three-stage workflow requires one tree worker")
    preparation = read_json(prepared / "metadata.json")
    if preparation.get("status") != "complete":
        raise ValueError("Preparation must be complete")
    prepared_hash = file_sha256(prepared / "metadata.json")
    prior = ExperimentConfig.model_validate(read_json(prepared / "configuration.json"))
    if any(getattr(prior, k) != getattr(config, k) for k in ("dataset", "preprocessing", "split")) or prior.seeds.split != config.seeds.split:
        raise ValueError("Search configuration differs from prepared data")
    for name in ("configuration.json", "preprocessing.json", "split_manifest.parquet"):
        if file_sha256(prepared / name) != preparation["files"][name]:
            raise ValueError("Preparation fingerprint mismatch")
    majority = preparation["splits"]["train"]["legitimate"]
    minority = preparation["splits"]["train"]["fraud"]
    if minority <= config.smote.k_neighbors or any(int(majority * r - minority) <= 0 for r in protocol.ratios):
        raise ValueError("Training class counts cannot support every declared SMOTE ratio")
    output = resume.resolve() if resume else new_run(output_root, "three_stage", run_id)
    with run_lock(output):
        plan_core = {"protocol": protocol.model_dump(mode="json"), "configuration": config.model_dump(),
                     "prepared_metadata_sha256": prepared_hash, "jobs": jobs}
        if resume:
            manifest = read_json(output / "metadata.json")
            if manifest.get("kind") != "validation_selection" or manifest.get("schema_version") != 2:
                raise ValueError("Resume requires a three-stage run")
            if file_sha256(output / "search_plan.json") != manifest["search_plan_sha256"]:
                raise ValueError("Frozen search plan fingerprint mismatch")
            plan = read_json(output / "search_plan.json")
            # Revision-1 plans written before explicit revision fields stay readable.
            comparable_plan = plan | {"protocol": SelectionProtocol.model_validate(plan["protocol"]).model_dump(mode="json")}
            if any(comparable_plan.get(k) != v for k, v in plan_core.items()):
                raise ValueError("Resume configuration differs from frozen search plan")
            for name, digest in manifest.get("frozen_stage_files", {}).items():
                if file_sha256(output / name) != digest:
                    raise ValueError("Frozen stage fingerprint mismatch")
            if manifest.get("status") == "complete":
                load_staged_selected(output)
                return output
        else:
            plan = plan_core | {
                "reuse": inventory(reuse_roots, output, config, prepared_hash, protocol),
                "implementation_sha256": {name: file_sha256(Path(__file__).with_name(name)) for name in
                     ("staged_selection.py", "threshold_search.py", "training.py", "evaluation.py", "research.py")},
                "package_versions": {name: version(name) for name in (*PACKAGES, "scipy", "matplotlib")}}
            atomic_json(output / "search_plan.json", plan)
            manifest = {"schema_version": 2, "kind": "validation_selection", "run_id": output.name,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "search_plan_sha256": file_sha256(output / "search_plan.json"),
                        "test_used": False, "stage": "three_stage_search"}
        manifest.update(status="running", worker_pid=os.getpid())
        manifest.setdefault("execution_history", []).append({
            "started_at": datetime.now(timezone.utc).isoformat(), "worker_pid": os.getpid(),
            "stop_after_stage": stop_after_stage,
            "selector_sha256": file_sha256(Path(__file__))})
        manifest.pop("failure", None)
        atomic_json(output / "metadata.json", manifest)
        completed = {}
        if (output / "search_progress.json").exists():
            completed = read_json(output / "search_progress.json")["completed_candidates"]
        models = list(plan["reuse"]["models"])
        reports = list(plan["reuse"]["reports"])
        for row in completed.values():
            models.append(row["model"])
            reports.append(row["report"])
        identity = None

        def candidate(ratio, forest):
            nonlocal identity
            key = candidate_key(ratio, forest)
            requested = candidate_config(config, ratio, forest)
            old = completed.get(key)
            model_path = None
            reusable_rf = None
            if old:
                model_path = resolve_reference(old["model"], output)
            else:
                # Recover fits completed just before interruption, even without a report checkpoint.
                extra = []
                for file in sorted((output / "candidates" / key).glob("*/metadata.json")):
                    if read_json(file).get("status") == "complete":
                        extra.append(reference(file.parent, output))
                for ref in extra + models:
                    source = resolve_reference(ref, output)
                    source_config = ExperimentConfig.model_validate(read_json(source / "configuration.json"))
                    if training_identity(source_config) == training_identity(requested):
                        inspect_model(source, prepared_hash)
                        model_path = source
                        break
                    if reusable_rf is None and training_identity(source_config, exclude_ratio=True) == training_identity(requested, exclude_ratio=True):
                        inspect_model(source, prepared_hash)
                        reusable_rf = source
                if model_path is None:
                    release_candidate_memory()
                    progress(f"Fitting {key}")
                    model_path = train_models(prepared, requested, output / "candidates" / key,
                                              jobs=jobs, progress=progress, reuse_rf_from=reusable_rf)
                    models.append(reference(model_path, output))
                else:
                    progress(f"Reusing {key}")
            _, actual_config = inspect_model(model_path, prepared_hash)
            if training_identity(actual_config) != training_identity(requested):
                raise ValueError("Candidate settings differ from search plan")
            report_path = resolve_reference(old["report"], output) if old else None
            if report_path is None:
                extra = [reference(p.parent, output) for p in sorted((output / "validation" / key).glob("*/metadata.json"))
                         if read_json(p).get("status") == "complete"]
                for ref in extra + reports:
                    path = resolve_reference(ref, output)
                    meta = read_json(path / "metadata.json")
                    if (meta.get("model_metadata_sha256") == file_sha256(model_path / "metadata.json")
                            and meta.get("policy") == config.evaluation.model_dump()):
                        report_path = path
                        break
            if report_path is None:
                release_candidate_memory()
                report_path = evaluate_run(model_path, prepared, config, output / "validation" / key, progress=progress)
                reports.append(reference(report_path, output))
            predictions = verify_report(report_path, model_path, config, prepared_hash)
            current = predictions[["transaction_id", "source_row_number", "actual_label", "split"]]
            if len(current) != preparation["splits"]["validation"]["rows"]:
                raise ValueError("Validation report row count differs from preparation")
            if identity is None:
                identity = current.copy()
            elif not identity.equals(current):
                raise ValueError("Validation identities or labels differ between candidates")
            row = summarize_candidate(predictions, requested) | {
                "key": key, "model": reference(model_path, output), "report": reference(report_path, output)}
            if any(saved["forest"] == row["forest"] and saved["rf"]["score_sha256"] != row["rf"]["score_sha256"]
                   for saved in completed.values()):
                raise ValueError("Benchmark RF validation scores differ across identical forest settings")
            if old and old != row:
                raise ValueError("Saved candidate metrics differ from recomputed validation metrics")
            completed[key] = row
            atomic_json(output / "search_progress.json", {"completed_candidates": completed})
            progress(f"Validated {key}: RF AP={row['rf']['pr_auc']:.6f}, RF-SMOTE AP={row['rf_smote']['pr_auc']:.6f}")
            del predictions, current
            release_candidate_memory()
            return row

        try:
            ratios = [candidate(r, protocol.reference_forest) for r in protocol.ratios]
            ratio_winner = choose_ratio(ratios)
            freeze_stage(output / "stages" / STAGES[0], ratios, ratio_winner, protocol.ratio_objective)
            progress(f"Stage 1 frozen: SMOTE ratio {ratio_winner['ratio']:.17g}")
            stage_names = [f"stages/{STAGES[0]}/{name}" for name in ("decision.json", "candidates.csv")]
            manifest["frozen_stage_files"] = {name: file_sha256(output / name) for name in stage_names}
            if stop_after_stage == 1:
                manifest.update(status="awaiting_next_stage", stage="ratio_selected", completed_stage=1,
                                stopped_at=datetime.now(timezone.utc).isoformat())
                atomic_json(output / "metadata.json", manifest)
                progress("Stage 1 complete; Stage 2, threshold selection, and test evaluation were not started")
                return output
            atomic_json(output / "metadata.json", manifest)
            forests = [candidate(ratio_winner["ratio"], forest) for forest in protocol.forests()]
            winner = choose_forest(forests)
            freeze_stage(output / "stages" / STAGES[1], forests, winner, protocol.forest_objective)
            progress(f"Stage 2 frozen: trees/depth/leaf {winner['forest']}")
            predictions = verify_report(resolve_reference(winner["report"], output),
                                        resolve_reference(winner["model"], output), config, prepared_hash)
            threshold, curve = all_score_thresholds(predictions.actual_label, predictions.rf_risk_score,
                                                     predictions.rf_smote_risk_score)
            direct = exact_mean_f1(predictions.actual_label, predictions.rf_risk_score,
                                   predictions.rf_smote_risk_score, threshold["threshold"])
            if direct != Fraction(threshold["mean_f1_fraction"]):
                raise ValueError("Threshold sweep differs from direct predictions")
            threshold_dir = output / "stages" / STAGES[2]
            threshold_dir.mkdir(parents=True, exist_ok=True)
            if (threshold_dir / "decision.json").exists() and read_json(threshold_dir / "decision.json") != threshold:
                raise ValueError("Frozen threshold differs from recomputed selection")
            atomic_json(threshold_dir / "decision.json", threshold)
            curve.to_parquet(threshold_dir / "thresholds.parquet", index=False)
            curve.to_csv(threshold_dir / "thresholds.csv", index=False, float_format="%.17g")
            save_threshold_figure(curve, threshold, threshold_dir / "thresholds.svg")
            selection = {"frozen": True, "selection_split": "validation", "test_used": False,
                         "protocol": protocol.model_dump(mode="json"), "ratio": ratio_winner["ratio"],
                         "forest": winner["forest"], "candidate_key": winner["key"],
                         "model": winner["model"], "validation_report": winner["report"], **threshold}
            atomic_json(output / "selection.json", selection)
            write_summary(output, ratio_winner, winner, threshold)
            names = ["search_plan.json", "search_progress.json", "selection.json", "SUMMARY.md"]
            names += [p.relative_to(output).as_posix() for p in (output / "stages").rglob("*") if p.is_file()]
            manifest.update(status="complete", stage="validation_selected",
                            files={name: file_sha256(output / name) for name in sorted(names)})
            atomic_json(output / "metadata.json", manifest)
            del predictions, curve, identity
            release_candidate_memory()
            load_staged_selected(output)
            progress(f"Three-stage selection frozen: ratio={selection['ratio']:.17g}, forest={selection['forest']}, threshold={selection['threshold']:.17g}")
            return output
        except BaseException as exc:
            manifest.update(status="failed", failure={"type": type(exc).__name__, "message": str(exc)})
            atomic_json(output / "metadata.json", manifest)
            raise


def save_threshold_figure(curve, decision, path):
    os.environ.setdefault("MPLCONFIGDIR", str(load_settings().backend_root / "runtime/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    indices = np.unique(np.r_[np.linspace(0, len(curve) - 1, min(3000, len(curve))).astype(int),
                              np.flatnonzero(curve.threshold == decision["threshold"])])
    sample = curve.iloc[indices]
    fig, ax = plt.subplots(figsize=(8, 5))
    for name in ("rf_f1", "rf_smote_f1", "mean_f1"):
        ax.plot(sample.threshold, sample[name], label=name)
    ax.axvline(decision["threshold"], color="black", linestyle="--", label="Selected cutoff")
    ax.set(xlabel="Shared fraud threshold", ylabel="Validation F1",
           title="Exact threshold selection (plot points sampled; full table retained)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_summary(output, ratio, forest, threshold):
    lines = ["# Three-stage validation selection", "", "Validation only; no held-out test evaluation.", "",
             f"- Stage 1: ratio {ratio['ratio']:.17g}; RF-SMOTE AP {ratio['rf_smote']['pr_auc']:.6f}.",
             f"- Stage 2: trees/depth/minimum leaf {forest['forest']}; mean AP {forest['mean_ap']:.6f}.",
             f"- Stage 3: common threshold {threshold['threshold']:.17g}; mean F1 {float(Fraction(threshold['mean_f1_fraction'])):.6f}.",
             "", "The protocol was revised after earlier validation experiments. Ratio selection uses a fixed",
             "reference forest; interactions with subsequently selected forest settings may be missed.",
             "Both models are retained. Settings were selected on one validation split and one set of seeds.",
             "Stages retain every candidate, the declared tie rules, full precision scores and provenance.", ""]
    (output / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def load_staged_selected(path):
    """Recompute every decision from its fingerprinted validation predictions."""
    meta = read_json(path / "metadata.json")
    required = {"search_plan.json", "search_progress.json", "selection.json", "SUMMARY.md",
                *(f"stages/{s}/{f}" for s in STAGES[:2] for f in ("decision.json", "candidates.csv")),
                *(f"stages/{STAGES[2]}/{f}" for f in ("decision.json", "thresholds.csv", "thresholds.parquet", "thresholds.svg"))}
    if (meta.get("schema_version") != 2 or meta.get("kind") != "validation_selection"
            or meta.get("status") != "complete" or meta.get("stage") != "validation_selected"
            or meta.get("test_used") is not False or set(meta.get("files", {})) != required):
        raise ValueError("Three-stage selection is incomplete or unsupported")
    for name, digest in meta["files"].items():
        if file_sha256(path / name) != digest:
            raise ValueError(f"Selected-run fingerprint mismatch: {name}")
    if meta["search_plan_sha256"] != meta["files"]["search_plan.json"]:
        raise ValueError("Selection search plan differs")
    plan = read_json(path / "search_plan.json")
    protocol = SelectionProtocol.model_validate(plan["protocol"])
    config = ExperimentConfig.model_validate(plan["configuration"])
    completed = read_json(path / "search_progress.json")["completed_candidates"]
    selection = read_json(path / "selection.json")
    if (selection.get("protocol") != plan["protocol"] or selection.get("frozen") is not True
            or selection.get("test_used") is not False or selection.get("selection_split") != "validation"):
        raise ValueError("Selected protocol or frozen state differs")
    identity = None
    selected_predictions = None
    benchmark_scores = {}
    for key, row in completed.items():
        model_path = resolve_reference(row["model"], path)
        _, actual = inspect_model(model_path, plan["prepared_metadata_sha256"])
        expected = candidate_config(config, row["ratio"], row["forest"])
        if training_identity(actual) != training_identity(expected) or key != candidate_key(row["ratio"], row["forest"]):
            raise ValueError("Selected candidate training configuration differs")
        predictions = verify_report(resolve_reference(row["report"], path), model_path, config, plan["prepared_metadata_sha256"])
        current = predictions[["transaction_id", "source_row_number", "actual_label", "split"]]
        if identity is None:
            identity = current.copy()
        elif not identity.equals(current):
            raise ValueError("Selected validation identities differ")
        calculated = summarize_candidate(predictions, expected)
        if any(row.get(k) != v for k, v in calculated.items()):
            raise ValueError("Selected candidate metrics differ from predictions")
        forest_key = tuple(row["forest"])
        digest = calculated["rf"]["score_sha256"]
        if benchmark_scores.setdefault(forest_key, digest) != digest:
            raise ValueError("Selected benchmark RF scores differ across ratios")
        if key == selection["candidate_key"]:
            selected_predictions = predictions
    decisions = [read_json(path / "stages" / s / "decision.json") for s in STAGES[:2]]
    ratios = [completed[candidate_key(r, protocol.reference_forest)] for r in protocol.ratios]
    ratio = choose_ratio(ratios)
    forests = [completed[candidate_key(ratio["ratio"], f)] for f in protocol.forests()]
    winner = choose_forest(forests)
    expected_keys = {r["key"] for r in ratios + forests}
    if set(completed) != expected_keys:
        raise ValueError("Unexpected candidates in sequential search")
    for decision, rows, chosen, objective in zip(decisions, (ratios, forests), (ratio, winner),
                                                  (protocol.ratio_objective, protocol.forest_objective)):
        if decision != {"frozen": True, "objective": objective, "candidates": rows, "chosen_key": chosen["key"]}:
            raise ValueError("Stage decision violates declared selection rule")
    if (selection["ratio"] != ratio["ratio"] or selection["forest"] != winner["forest"]
            or selection["candidate_key"] != winner["key"] or selection["model"] != winner["model"]
            or selection["validation_report"] != winner["report"] or selected_predictions is None):
        raise ValueError("Final selection differs from stage decisions")
    threshold, curve = all_score_thresholds(selected_predictions.actual_label, selected_predictions.rf_risk_score,
                                           selected_predictions.rf_smote_risk_score)
    if (read_json(path / "stages" / STAGES[2] / "decision.json") != threshold
            or any(selection.get(k) != v for k, v in threshold.items())):
        raise ValueError("Threshold selection violates declared rule")
    pd.testing.assert_frame_equal(curve, pd.read_parquet(path / "stages" / STAGES[2] / "thresholds.parquet"), check_exact=True)
    del predictions, selected_predictions, curve, identity, current
    release_candidate_memory()
    original = load_bundle(resolve_reference(winner["model"], path))
    effective = original.config.model_copy(deep=True)
    effective.scoring.threshold = threshold["threshold"]
    effective.random_forest.tuning_procedure = "three_stage_validation_selected"
    return ModelBundle(original.models, original.preprocessor, effective,
                       original.metadata | {"stage": "validation_selected", "run_id": meta["run_id"],
                                            "base_model_run_id": original.metadata["run_id"], "selection": selection})


def export_validation(selection_path, prepared, config, reports_root, progress=print):
    """Recoverable final report export; never requests the held-out test split."""
    root = reports_root / selection_path.name
    root.mkdir(parents=True, exist_ok=True)
    target = root / "final_validation"
    # A report may finish before the review manifest is written. Recover it too.
    targets = [target] + sorted(p.parent for p in root.glob("evaluate_*/metadata.json"))
    existing = next((p for p in targets if (p / "metadata.json").exists()
                     and read_json(p / "metadata.json").get("status") == "complete"), None)
    if existing is not None:
        target = existing
        meta, _ = load_report(target)
        if (meta["model_metadata_sha256"] != file_sha256(selection_path / "metadata.json")
                or meta["split"] != "validation" or meta["official_test"] is not False
                or meta["prepared_metadata_sha256"] != file_sha256(prepared / "metadata.json")
                or meta["policy"] != config.evaluation.model_dump()):
            raise ValueError("Existing final validation report differs")
    else:
        # Failed reports are retained; new attempts receive a unique run name.
        target = evaluate_run(selection_path, prepared, config, root,
                              run_id=None if target.exists() else "final_validation", progress=progress)
    import shutil
    for name in ("SUMMARY.md", "selection.json", "search_plan.json"):
        shutil.copyfile(selection_path / name, root / name)
    for stage in STAGES:
        destination = root / "stages" / stage
        destination.mkdir(parents=True, exist_ok=True)
        for source in (selection_path / "stages" / stage).iterdir():
            shutil.copyfile(source, destination / source.name)
    files = {p.relative_to(root).as_posix(): file_sha256(p) for p in root.rglob("*")
             if p.is_file() and p != root / "metadata.json"}
    atomic_json(root / "metadata.json", {"status": "complete", "kind": "three_stage_validation_review",
                 "test_used": False, "selection": str(selection_path),
                 "selection_metadata_sha256": file_sha256(selection_path / "metadata.json"),
                 "final_validation": os.path.relpath(target, root), "files": files})
    return root


def export_ratio_stage(selection_path, reports_root):
    """Publish the frozen ratio comparison without claiming a final selection."""
    import shutil
    manifest = read_json(selection_path / "metadata.json")
    for name, digest in manifest.get("frozen_stage_files", {}).items():
        if file_sha256(selection_path / name) != digest:
            raise ValueError("Frozen stage fingerprint mismatch")
    source = selection_path / "stages" / STAGES[0]
    decision = read_json(source / "decision.json")
    plan = read_json(selection_path / "search_plan.json")
    if file_sha256(selection_path / "search_plan.json") != manifest["search_plan_sha256"]:
        raise ValueError("Frozen search plan fingerprint mismatch")
    winner = choose_ratio(decision["candidates"])
    if not decision["frozen"] or winner["key"] != decision["chosen_key"]:
        raise ValueError("Invalid ratio stage decision")
    root = reports_root / selection_path.name / "stage1_review"
    root.mkdir(parents=True, exist_ok=True)
    for name in ("decision.json", "candidates.csv"):
        shutil.copyfile(source / name, root / name)
    shutil.copyfile(selection_path / "search_plan.json", root / "search_plan.json")
    lines = ["# Stage 1: SMOTE ratio validation", "", "Validation only; held-out test unused.", "",
             f"Protocol revision: {plan['protocol'].get('revision', 1)}.",
             *([plan['protocol']['amendment_note'], ""] if plan['protocol'].get('amendment_note') else []),
             f"Reference forest (trees/depth/minimum leaf): {plan['protocol']['reference_forest']}.",
             "Selection: highest RF-SMOTE Average Precision; exact ties prefer the smaller ratio.", "",
             "| Fraud/legitimate ratio | Benchmark RF AP | RF-SMOTE AP | Selected |",
             "|---|---:|---:|---|"]
    for row in decision["candidates"]:
        lines.append(f"| {row['ratio']:.17g} | {row['rf']['pr_auc']:.9f} | {row['rf_smote']['pr_auc']:.9f} | "
                     f"{'Yes' if row['key'] == winner['key'] else ''} |")
    lines += ["", f"Frozen ratio: {winner['ratio']:.17g}.",
              "Supplementary classification metrics in candidates.csv use 0.50; this is not a selected cutoff.",
              "This report does not select a forest configuration, final threshold, or model winner.",
              "The protocol was revised after earlier validation results; sequential selection can miss interactions.", ""]
    (root / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    atomic_json(root / "metadata.json", {"status": "complete", "kind": "stage1_ratio_review",
                 "test_used": False, "selection_run": str(selection_path),
                 "selection_metadata_sha256": file_sha256(selection_path / "metadata.json"),
                 "selected_ratio": winner["ratio"],
                 "files": {name: file_sha256(root / name) for name in
                           ("decision.json", "candidates.csv", "search_plan.json", "SUMMARY.md")}})
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--protocol", type=Path, default=Path("configs/validation_three_stage.yaml"))
    parser.add_argument("--run-id")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--stop-after-stage", type=int, choices=(1, 3), default=3,
                        help="Use 1 to freeze only the SMOTE ratio and stop before forest tuning")
    args = parser.parse_args()
    try:
        settings = load_settings()
        resolve = lambda p: p if p.is_absolute() else settings.backend_root / p
        config = load_experiment(resolve(args.config or settings.experiment_config))
        prepared = resolve(args.prepared)
        progress = lambda message: print(message, file=sys.stderr, flush=True)
        selected = select_three_stage(prepared, config, load_protocol(resolve(args.protocol)), settings.artifacts_dir,
                                      args.run_id, args.jobs, resolve(args.resume) if args.resume else None,
                                      (settings.artifacts_dir, settings.reports_dir), progress,
                                      stop_after_stage=args.stop_after_stage)
        report = (export_ratio_stage(selected, settings.reports_dir) if args.stop_after_stage == 1 else
                  export_validation(selected, prepared, config, settings.reports_dir, progress))
    except (ValueError, OSError, MemoryError) as exc:
        print(json.dumps({"success": False, "error": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps({"success": True, "selection": str(selected), "report": str(report)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
