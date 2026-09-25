"""Predeclared validation search and immutable selected-run references."""
import argparse
from datetime import datetime, timezone
from fractions import Fraction
import gc
import json
import os
from pathlib import Path
import sys
import numpy as np
import pyarrow as pa
from integritree.config import load_experiment, ExperimentConfig
from integritree.ml.data import file_sha256, write_json
from integritree.ml.evaluation import binary_labels, score_vector
from integritree.ml.research import evaluate_run, load_report, new_run
from integritree.settings import load_settings

CANDIDATES = ((100,10), (100,20), (200,10), (200,20))
THRESHOLDS = tuple(i / 20 for i in range(1,20))
PROTOCOL = {"candidates": [list(c) for c in CANDIDATES],
            "configuration_threshold": .5, "objective": "mean_validation_f1",
            "configuration_ties": "shallower_then_fewer_trees", "thresholds": list(THRESHOLDS),
            "threshold_ties": "closest_to_0.50_then_higher", "selection_split": "validation"}


def exact_mean_f1(y, rf, smote, threshold):
    y = binary_labels(y)
    if set(y.tolist()) != {0,1}:
        raise ValueError("Selection requires both ground-truth classes")
    values = []
    for scores in (rf, smote):
        pred = score_vector(scores, len(y)) >= threshold
        tp = int(((y == 1) & pred).sum())
        fp = int(((y == 0) & pred).sum())
        fn = int(((y == 1) & ~pred).sum())
        values.append(Fraction(2 * tp, 2 * tp + fp + fn))
    return sum(values) / 2


def choose_configuration(rows):
    if len(rows) != 4 or {tuple(row["candidate"]) for row in rows} != set(CANDIDATES):
        raise ValueError("Exactly the four approved RF candidates are required")
    return max(rows, key=lambda r: (Fraction(r["mean_f1_fraction"]), -r["candidate"][1], -r["candidate"][0]))


def choose_threshold(y, rf, smote):
    rows = [{"threshold": t, "mean_f1_fraction": str(exact_mean_f1(y, rf, smote, t))}
            for t in THRESHOLDS]
    # Integer twentieths avoid binary floating-point tie-break asymmetry.
    best = max(rows, key=lambda r: (Fraction(r["mean_f1_fraction"]),
                                   -abs(round(r["threshold"]*20)-10), round(r["threshold"]*20)))
    return best["threshold"], rows


def same_training(base, candidate):
    return (base.dataset == candidate.dataset and base.preprocessing == candidate.preprocessing
            and base.split == candidate.split and base.seeds == candidate.seeds
            and base.smote == candidate.smote
            and base.random_forest.model_dump(exclude={"n_estimators","max_depth"})
                == candidate.random_forest.model_dump(exclude={"n_estimators","max_depth"})
            and base.scoring == candidate.scoring)


def release_candidate_memory():
    """Return unused report/Arrow allocations before the next training preflight.

    This is best-effort allocator cleanup, not a reduction of the training data
    or a bypass of the existing RAM safety check.
    """
    gc.collect()
    pa.default_memory_pool().release_unused()


def select_models(baseline_path, prepared, config, output_root, run_id=None,
                  jobs=1, resume=None, progress=print):
    from integritree.ml.artifacts import load_bundle
    from integritree.ml.training import train_models
    config.require_ready("evaluate")
    base = load_bundle(baseline_path)
    if base.metadata.get("stage") != "baseline_not_final" or not same_training(base.config, config):
        raise ValueError("Search requires compatible immutable baseline training settings")
    if (base.config.random_forest.n_estimators, base.config.random_forest.max_depth) != (100,20):
        raise ValueError("The approved baseline is 100 trees, depth 20")
    from integritree.ml.research import verify_prepared
    verify_prepared(base, prepared)
    baseline_hash = file_sha256(baseline_path / "metadata.json")
    prepared_hash = file_sha256(prepared / "metadata.json")
    output = resume.resolve() if resume else new_run(output_root, "selection", run_id)
    if resume:
        previous = json.loads((output / "metadata.json").read_text())
        if previous.get("status") == "complete" or previous.get("kind") != "validation_selection":
            raise ValueError("Only an incomplete selection can be resumed")
        plan = json.loads((output / "search_plan.json").read_text())
        expected = {"protocol": PROTOCOL, "baseline_sha256": baseline_hash,
                    "prepared_sha256": prepared_hash, "configuration": config.model_dump()}
        if plan != expected:
            raise ValueError("Resume configuration/provenance differs from frozen search plan")
    else:
        write_json(output / "search_plan.json", {"protocol": PROTOCOL, "baseline_sha256": baseline_hash,
                   "prepared_sha256": prepared_hash, "configuration": config.model_dump()})
    manifest = {"schema_version": 1, "kind": "validation_selection", "stage": "validation_selected",
                "run_id": output.name, "status": "running", "created_at": datetime.now(timezone.utc).isoformat()}
    write_json(output / "metadata.json", manifest)
    del base
    release_candidate_memory()
    try:
        rows, reference_ids, reference_labels = [], None, None
        for trees, depth in CANDIDATES:
            key = f"trees_{trees}_depth_{depth}"
            candidate_config = config.model_copy(deep=True)
            candidate_config.random_forest.n_estimators = trees
            candidate_config.random_forest.max_depth = depth
            if (trees,depth) == (100,20):
                candidate_path = baseline_path
            else:
                root = output / "candidates" / key
                completed = sorted(p.parent for p in root.glob("*/metadata.json")
                                   if json.loads(p.read_text()).get("status") == "complete")
                if completed:
                    candidate_path = completed[-1]
                else:
                    progress(f"Training approved candidate {key}")
                    candidate_path = train_models(prepared, candidate_config, root, jobs=jobs, progress=progress)
            candidate = load_bundle(candidate_path)
            if not same_training(config, candidate.config) or (
                    candidate.config.random_forest.n_estimators, candidate.config.random_forest.max_depth) != (trees,depth):
                raise ValueError("Candidate settings differ from the approved search")
            del candidate
            release_candidate_memory()
            report_root = output / "validation" / key
            reports = sorted(p.parent for p in report_root.glob("*/metadata.json")
                             if json.loads(p.read_text()).get("status") == "complete")
            report_path = reports[-1] if reports else evaluate_run(
                candidate_path, prepared, config, report_root, progress=progress)
            report_meta, predictions = load_report(report_path)
            if report_meta["split"] != "validation" or report_meta["model_metadata_sha256"] != file_sha256(candidate_path / "metadata.json"):
                raise ValueError("Candidate validation report is incompatible")
            ids, y = predictions.transaction_id.to_numpy(), predictions.actual_label.to_numpy()
            if reference_ids is None:
                reference_ids, reference_labels = ids, y
            elif not np.array_equal(reference_ids, ids) or not np.array_equal(reference_labels, y):
                raise ValueError("Candidate validation identities/labels differ")
            f1 = exact_mean_f1(y, predictions.rf_risk_score, predictions.rf_smote_risk_score, .5)
            rows.append({"candidate": [trees,depth], "mean_f1": float(f1), "mean_f1_fraction": str(f1),
                         "models": os.path.relpath(candidate_path, output),
                         "model_metadata_sha256": file_sha256(candidate_path / "metadata.json"),
                         "report": os.path.relpath(report_path, output),
                         "report_metadata_sha256": file_sha256(report_path / "metadata.json")})
            write_json(output / "search_progress.json", {"completed_candidates": rows})
            # Retain only the first report's identity/label arrays for pairing.
            # The full previous report must not survive into the next fit.
            del predictions, ids, y
            release_candidate_memory()
        winner = choose_configuration(rows)
        _, predictions = load_report(output / winner["report"])
        threshold, threshold_rows = choose_threshold(predictions.actual_label, predictions.rf_risk_score,
                                                     predictions.rf_smote_risk_score)
        selection = {"frozen": True, "selection_split": "validation", "test_used": False,
                     "prepared_metadata_sha256": prepared_hash, "baseline_metadata_sha256": baseline_hash,
                     "protocol": PROTOCOL, "candidates": rows, "chosen_candidate": winner["candidate"],
                     "model_path": winner["models"], "model_metadata_sha256": winner["model_metadata_sha256"],
                     "threshold": threshold, "threshold_search": threshold_rows,
                     "threshold_mean_f1_fraction": next(r["mean_f1_fraction"] for r in threshold_rows if r["threshold"] == threshold)}
        write_json(output / "selection.json", selection)
        manifest.update(status="complete", files={name: file_sha256(output / name)
                         for name in ("search_plan.json","selection.json","search_progress.json")})
        write_json(output / "metadata.json", manifest)
        load_selected(output)
        progress(f"Selection frozen: {winner['candidate']}, shared cutoff {threshold:.2f}")
        return output
    except BaseException as exc:
        manifest.update(status="failed", failure_type=type(exc).__name__)
        write_json(output / "metadata.json", manifest)
        raise


def load_selected(path):
    from integritree.ml.artifacts import load_bundle, ModelBundle
    meta = json.loads((path / "metadata.json").read_text())
    if meta.get("status") != "complete" or meta.get("kind") != "validation_selection" or meta.get("schema_version") != 1:
        raise ValueError("Selected run is incomplete or unsupported")
    if set(meta.get("files", {})) != {"search_plan.json","selection.json","search_progress.json"}:
        raise ValueError("Selected-run manifest is incomplete")
    for name, digest in meta["files"].items():
        if file_sha256(path / name) != digest:
            raise ValueError("Selected-run fingerprint mismatch")
    selection = json.loads((path / "selection.json").read_text())
    if (selection.get("protocol") != PROTOCOL or not selection.get("frozen")
            or selection.get("test_used") is not False or selection.get("selection_split") != "validation"):
        raise ValueError("Selection protocol or frozen state is invalid")
    plan = json.loads((path / "search_plan.json").read_text())
    if (plan["protocol"] != PROTOCOL or plan["prepared_sha256"] != selection["prepared_metadata_sha256"]
            or plan["baseline_sha256"] != selection["baseline_metadata_sha256"]):
        raise ValueError("Selection differs from the frozen search plan")
    search_config = ExperimentConfig.model_validate(plan["configuration"])
    winner = choose_configuration(selection["candidates"])
    if winner["candidate"] != selection["chosen_candidate"] or winner["models"] != selection["model_path"]:
        raise ValueError("Selected candidate violates the selection rule")
    # Validate actual recorded validation predictions and tie-breaking on load.
    reference = None
    selected_predictions = None
    for row in selection["candidates"]:
        candidate_path = (path / row["models"]).resolve()
        if file_sha256(candidate_path / "metadata.json") != row["model_metadata_sha256"]:
            raise ValueError("Candidate model provenance mismatch")
        candidate_meta = json.loads((candidate_path / "metadata.json").read_text())
        if (candidate_meta.get("status") != "complete" or candidate_meta.get("stage") != "baseline_not_final"
                or candidate_meta.get("prepared_metadata_sha256") != selection["prepared_metadata_sha256"]
                or file_sha256(candidate_path / "configuration.json") != candidate_meta["files"]["configuration.json"]):
            raise ValueError("Candidate training provenance mismatch")
        candidate_config = ExperimentConfig.model_validate_json((candidate_path / "configuration.json").read_text())
        if (not same_training(search_config, candidate_config)
                or [candidate_config.random_forest.n_estimators, candidate_config.random_forest.max_depth] != row["candidate"]):
            raise ValueError("Candidate configuration differs from approved selection")
        report_path = (path / row["report"]).resolve()
        if file_sha256(report_path / "metadata.json") != row["report_metadata_sha256"]:
            raise ValueError("Selection report provenance mismatch")
        report_meta, predictions = load_report(report_path)
        if report_meta["split"] != "validation" or report_meta["prepared_metadata_sha256"] != selection["prepared_metadata_sha256"]:
            raise ValueError("Selection requires the shared validation split")
        if report_meta["model_metadata_sha256"] != row["model_metadata_sha256"]:
            raise ValueError("Selection model/report mismatch")
        identity = predictions[["transaction_id","actual_label"]]
        if reference is None:
            reference = identity
        elif not reference.equals(identity):
            raise ValueError("Selection validation pairing differs")
        if exact_mean_f1(predictions.actual_label, predictions.rf_risk_score, predictions.rf_smote_risk_score, .5) != Fraction(row["mean_f1_fraction"]):
            raise ValueError("Selection F1 does not match saved validation predictions")
        if row is winner:
            selected_predictions = predictions
    threshold, curve = choose_threshold(selected_predictions.actual_label,
                                        selected_predictions.rf_risk_score, selected_predictions.rf_smote_risk_score)
    if threshold != selection["threshold"] or curve != selection["threshold_search"]:
        raise ValueError("Selected threshold violates the validation rule")
    model_path = (path / selection["model_path"]).resolve()
    if file_sha256(model_path / "metadata.json") != selection["model_metadata_sha256"]:
        raise ValueError("Selected model fingerprint mismatch")
    model_meta = json.loads((model_path / "metadata.json").read_text())
    if model_meta.get("stage") != "baseline_not_final":
        raise ValueError("Selected-run references must point to original trained bundles")
    original = load_bundle(model_path)
    if [original.config.random_forest.n_estimators, original.config.random_forest.max_depth] != winner["candidate"]:
        raise ValueError("Chosen model settings differ from the selected candidate")
    if original.metadata["prepared_metadata_sha256"] != selection["prepared_metadata_sha256"]:
        raise ValueError("Selected preparation provenance mismatch")
    selected_config = original.config.model_copy(deep=True)
    selected_config.scoring.threshold = threshold
    selected_config.random_forest.tuning_procedure = "validation_grid_selected"
    selected_meta = original.metadata | {"stage": "validation_selected", "run_id": meta["run_id"],
                                        "base_model_run_id": original.metadata["run_id"],
                                        "selection": selection}
    return ModelBundle(original.models, original.preprocessor, selected_config, selected_meta)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    try:
        settings = load_settings()
        resolve = lambda p: p if p.is_absolute() else settings.backend_root / p
        config = load_experiment(resolve(args.config or settings.experiment_config))
        output = select_models(resolve(args.baseline), resolve(args.prepared), config, settings.artifacts_dir,
                               args.run_id, args.jobs, resolve(args.resume) if args.resume else None,
                               lambda x: print(x, file=sys.stderr, flush=True))
    except (ValueError, OSError, MemoryError) as exc:
        print(json.dumps({"success": False,"error": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps({"success":True,"selection":str(output)}))
    return 0
