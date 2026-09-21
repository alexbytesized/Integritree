"""Train a matched RF/RF-SMOTE baseline from a completed preparation bundle."""
import argparse
import ctypes
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import re
import shutil
import sys
import time
import uuid
from imblearn.over_sampling import SMOTE
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from integritree.config import ExperimentConfig, load_experiment
from integritree.ml.artifacts import PACKAGES, REQUIRED_FILES, load_bundle
from integritree.ml.data import file_sha256, write_json
from integritree.ml.features import FEATURE_COLUMNS, SCALED_COLUMNS
from integritree.ml.inference import feature_matrix, fraud_scores
from integritree.ml.preparation import load_prepared_split, validate_preparation_config
from integritree.ml.preprocessing import FittedPreprocessor
from integritree.settings import load_settings


def validate_training_config(config: ExperimentConfig) -> None:
    config.require_ready("train")
    validate_preparation_config(config)
    expected = {"smote.method": "smote_encoded", "smote.sampling_ratio": 1.0,
                "scoring.probability_method": "mean_tree_probability", "scoring.tie_policy": "fraud",
                "scoring.threshold": 0.5,
                "random_forest.tuning_procedure": "fixed_baseline"}
    for field, value in expected.items():
        section, name = field.split(".")
        if getattr(getattr(config, section), name) != value:
            raise ValueError(f"Unsupported Phase 3 setting {field}; baseline training only")


def memory_preflight(rows: int, majority: int) -> dict:
    """Conservative working-array estimate, not a guarantee of RF peak memory."""
    estimate = (3 * rows + 4 * (2 * majority)) * len(FEATURE_COLUMNS) * 8 + 64 * 1024**2
    available = None
    if sys.platform == "win32":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong) for name in (
                    "total", "available", "page_total", "page_available",
                    "virtual_total", "virtual_available", "extended")]
        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            available = status.available
    if available is not None and available < estimate:
        raise MemoryError(f"Insufficient available RAM: {available / 1024**3:.2f} GiB; "
                          f"estimated working arrays need {estimate / 1024**3:.2f} GiB "
                          "before additional tree/runtime memory. Free memory before retrying.")
    return {"estimated_array_bytes": estimate, "available_bytes": available}


def forest_parameters(config: ExperimentConfig, jobs: int) -> dict:
    if isinstance(jobs, bool) or not isinstance(jobs, int) or jobs < 1:
        raise ValueError("jobs must be a positive integer")
    rf = config.random_forest
    return {"n_estimators": rf.n_estimators,
            "max_depth": None if rf.max_depth == "unlimited" else rf.max_depth,
            "min_samples_split": rf.min_samples_split, "min_samples_leaf": rf.min_samples_leaf,
            "max_features": None if rf.max_features == "all" else rf.max_features,
            "bootstrap": rf.bootstrap, "class_weight": None, "criterion": "gini",
            "random_state": config.seeds.model, "n_jobs": jobs}


def validate_labels(labels, rows: int, neighbors: int) -> np.ndarray:
    values = np.asarray(labels)
    if values.ndim != 1 or len(values) != rows or not np.isin(values, [0, 1]).all():
        raise ValueError("Training labels must be aligned binary 0/1 values")
    counts = np.bincount(values.astype("uint8"), minlength=2)
    if counts[1] <= neighbors:
        raise ValueError("Too few fraud training records for smote.k_neighbors; need k + 1")
    if counts[1] >= counts[0]:
        raise ValueError("Fraud must be the minority class in the original training set")
    return values.astype("uint8")


def resample_training(values: np.ndarray, labels: np.ndarray, config: ExperimentConfig):
    # Explicit float conversion prevents integer dtype truncation by resampling.
    values = np.asarray(values, dtype="float64")
    sampler = SMOTE(sampling_strategy=config.smote.sampling_ratio,
                    k_neighbors=config.smote.k_neighbors, random_state=config.seeds.smote)
    resampled, targets = sampler.fit_resample(values, labels)
    if not np.array_equal(resampled[:len(values)], values) or not np.array_equal(targets[:len(labels)], labels):
        raise ValueError("SMOTE did not preserve original training rows")
    if not np.isfinite(resampled).all() or not np.all(targets[len(labels):] == 1):
        raise ValueError("Invalid synthetic features or labels")
    counts = np.bincount(targets, minlength=2)
    if counts[0] != counts[1]:
        raise ValueError("SMOTE did not produce the requested 1:1 balance")
    indicators = [i for i, name in enumerate(FEATURE_COLUMNS) if name not in SCALED_COLUMNS]
    fractional = {FEATURE_COLUMNS[i]: 0 for i in indicators}
    mixed_type_rows = 0
    type_indices = [i for i, name in enumerate(FEATURE_COLUMNS) if name.startswith("type_")]
    for start in range(len(values), len(resampled), 100000):
        batch = resampled[start:start + 100000]
        for i in indicators:
            fractional[FEATURE_COLUMNS[i]] += int(((batch[:, i] != 0) & (batch[:, i] != 1)).sum())
        mixed_type_rows += int(((batch[:, type_indices] > 0).sum(axis=1) > 1).sum())
    audit = {"original_rows": len(values), "synthetic_rows": len(resampled) - len(values),
             "resampled_rows": len(resampled), "class_counts": {"0": int(counts[0]), "1": int(counts[1])},
             "dtype": str(resampled.dtype), "fractional_indicator_counts": fractional,
             "multiple_positive_type_rows": mixed_type_rows, "synthetic_label": 1,
             "fraction_policy": "preserve; no rounding, argmax, truncation, or filtering"}
    return resampled, targets, audit


def train_models(prepared: Path, config: ExperimentConfig, output_root: Path,
                 run_id: str | None = None, jobs: int = 1, progress=print) -> Path:
    validate_training_config(config)
    parameters = forest_parameters(config, jobs)
    preparation = json.loads((prepared / "metadata.json").read_text(encoding="utf-8"))
    if preparation.get("status") != "complete":
        raise ValueError("Preparation bundle is incomplete")
    for name in ("configuration.json", "preprocessing.json"):
        if file_sha256(prepared / name) != preparation["files"][name]:
            raise ValueError(f"Preparation fingerprint mismatch: {name}")
    prior = ExperimentConfig.model_validate_json((prepared / "configuration.json").read_text())
    if (prior.dataset != config.dataset or prior.preprocessing != config.preprocessing
            or prior.split != config.split or prior.seeds.split != config.seeds.split):
        raise ValueError("Training configuration does not match prepared dataset/preprocessing/splits")
    resources = memory_preflight(preparation["splits"]["train"]["rows"],
                                 preparation["splits"]["train"]["legitimate"])
    preprocessor = FittedPreprocessor.load(prepared / "preprocessing.json")
    progress("Loading and verifying original training split only")
    features, labels, ids = load_prepared_split(prepared, "train")
    values = feature_matrix(features)
    labels = validate_labels(labels, len(values), config.smote.k_neighbors)
    if preprocessor.state.training_rows != len(values):
        raise ValueError("Preprocessor training count does not match training split")
    del features, ids
    run_id = run_id or ("train_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8])
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,100}", run_id):
        raise ValueError("Invalid run_id")
    output = output_root.resolve() / run_id
    output.mkdir(parents=True, exist_ok=False)
    metadata = {"schema_version": 1, "status": "running", "run_id": run_id,
                "created_at": datetime.now(timezone.utc).isoformat(), "stage": "baseline_not_final",
                "feature_order": FEATURE_COLUMNS, "dataset": config.dataset.model_dump(),
                "prepared_run_id": preparation["run_id"],
                "prepared_metadata_sha256": file_sha256(prepared / "metadata.json"),
                "split_manifest_sha256": preparation["split_manifest_sha256"],
                "package_versions": {p: version(p) for p in PACKAGES},
                "test_used": False, "validation_used": False, "rf_parameters": parameters,
                "resource_preflight": resources}
    write_json(output / "metadata.json", metadata)
    started = time.monotonic()
    try:
        write_json(output / "configuration.json", config.model_dump())
        shutil.copyfile(prepared / "preprocessing.json", output / "preprocessing.json")
        shutil.copyfile(prepared / "metadata.json", output / "prepared_metadata.json")
        reference = values[:min(128, len(values))].copy()
        verification = {"source": "first up to 128 original training feature rows",
                        "rows": len(reference), "absolute_tolerance": 1e-12, "models": {}}
        for name in ("rf", "rf_smote"):
            if name == "rf_smote":
                progress("Applying ordinary SMOTE to training only, preserving synthetic fractions")
                values, labels, audit = resample_training(values, labels, config)
                write_json(output / "training_audit.json", audit)
            progress(f"Training {name}: {len(values):,} rows, {parameters['n_estimators']} trees")
            model = RandomForestClassifier(**parameters).fit(values, labels)
            expected = fraud_scores(model, reference)
            joblib.dump(model, output / f"{name}.joblib", compress=3)
            del model
            reloaded = joblib.load(output / f"{name}.joblib")
            actual = fraud_scores(reloaded, reference)
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
            verification["models"][name] = {"passed": True, "max_absolute_error": float(np.max(np.abs(actual - expected)))}
            del reloaded
            progress(f"Saved and verified {name}")
        del values, labels
        write_json(output / "reload_verification.json", verification)
        metadata["implementation_sha256"] = {p.name: file_sha256(p) for p in (
            Path(__file__), Path(__file__).with_name("inference.py"), Path(__file__).with_name("artifacts.py"))}
        metadata["files"] = {name: file_sha256(output / name) for name in sorted(REQUIRED_FILES)}
        metadata.update(status="complete", elapsed_seconds=time.monotonic() - started)
        write_json(output / "metadata.json", metadata)
        load_bundle(output)
        progress("Baseline bundle complete; validation tuning and final test evaluation have not run")
        return output
    except BaseException as exc:
        metadata.update(status="failed", failure_type=type(exc).__name__)
        write_json(output / "metadata.json", metadata)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True, help="Completed preparation bundle, relative to backend root")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--jobs", type=int, default=1, help="Parallel tree workers; keep low for large datasets")
    args = parser.parse_args()
    try:
        settings = load_settings()
        resolve = lambda p: p if p.is_absolute() else settings.backend_root / p
        config = load_experiment(resolve(args.config or settings.experiment_config))
        output = train_models(resolve(args.prepared), config, settings.artifacts_dir,
                              args.run_id, args.jobs,
                              progress=lambda message: print(message, file=sys.stderr, flush=True))
    except (ValueError, OSError, MemoryError) as exc:
        print(json.dumps({"success": False, "error": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps({"success": True, "bundle": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
