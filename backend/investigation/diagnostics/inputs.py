"""Read verified validation evidence without changing any source artifact."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

from integritree.config import EvaluationConfig
from integritree.ml.artifacts import load_bundle
from integritree.ml.data import file_sha256
from integritree.ml.evaluation import metrics, score_vector
from integritree.ml.features import FEATURE_COLUMNS
from integritree.ml.preparation import load_prepared_split
from integritree.ml.research import load_report, verify_prepared


def read_config(path):
    path = Path(path).resolve()
    settings = yaml.safe_load(path.read_text(encoding="utf-8"))
    allowed = {
        "schema_version",
        "models",
        "prepared",
        "report",
        "sample_per_group",
        "seed",
        "minimum_denominator",
    }
    if (
        not isinstance(settings, dict)
        or set(settings) != allowed
        or settings["schema_version"] != 1
    ):
        raise ValueError("Unsupported investigation configuration fields/schema")
    for key in ("sample_per_group", "seed", "minimum_denominator"):
        value = settings[key]
        if type(value) is not int or value < (0 if key == "seed" else 1):
            raise ValueError(f"Invalid {key}")
    for key in ("models", "prepared", "report"):
        if not isinstance(settings[key], str) or not settings[key].strip():
            raise ValueError(f"Missing {key} path")
        settings[key] = (path.parent / settings[key]).resolve()
    return settings


def verify_links(meta, bundle, model_path, prepared):
    if meta.get("split") != "validation" or meta.get("official_test") is not False:
        raise ValueError("Investigation accepts validation reports only")
    expected = {
        "model_metadata_sha256": file_sha256(model_path / "metadata.json"),
        "prepared_metadata_sha256": file_sha256(prepared / "metadata.json"),
        "dataset_sha256": bundle.config.dataset.sha256,
        "model_run_id": bundle.metadata["run_id"],
        "split_manifest_sha256": file_sha256(prepared / "split_manifest.parquet"),
    }
    for name, value in expected.items():
        if meta.get(name) != value:
            raise ValueError(f"Report provenance mismatch: {name}")
    if meta.get("threshold") != bundle.config.scoring.threshold:
        raise ValueError("Report/model threshold mismatch")


def align_predictions(predictions, rows, labels, dataset_hash, run_id, threshold):
    """Verify exact validation membership; permit ordering differences, never guessing."""
    p = predictions.copy()
    required = {
        "source_row_number",
        "transaction_id",
        "actual_label",
        "split",
        "run_id",
        "threshold",
        "rf_risk_score",
        "rf_smote_risk_score",
        "rf_predicted_label",
        "rf_smote_predicted_label",
    }
    if not required.issubset(p.columns) or len(p) != len(rows):
        raise ValueError("Missing prediction fields or validation rows")
    if p[list(required)].isna().any().any():
        raise ValueError("Missing prediction identities/values")
    if (
        not p.source_row_number.is_unique
        or not p.transaction_id.is_unique
        or set(p.source_row_number) != set(rows)
    ):
        raise ValueError("Duplicate or mismatched validation identities")
    p = p.set_index("source_row_number").loc[list(rows)].reset_index()
    if not p.transaction_id.tolist() == [f"{dataset_hash}:{int(row)}" for row in rows]:
        raise ValueError("Transaction identities do not match dataset/row identities")
    if (
        not p.split.eq("validation").all()
        or not p.run_id.eq(run_id).all()
        or not p.threshold.eq(threshold).all()
    ):
        raise ValueError("Prediction split/run/threshold mismatch")
    if not np.array_equal(p.actual_label.to_numpy(), np.asarray(labels)):
        raise ValueError("Prediction labels differ from validation membership")
    if set(p.actual_label.unique()) != {0, 1}:
        raise ValueError("Validation diagnostics require both classes")
    for model in ("rf", "rf_smote"):
        score = score_vector(p[f"{model}_risk_score"], len(p))
        if not np.array_equal(
            p[f"{model}_predicted_label"], (score >= threshold).astype(int)
        ):
            raise ValueError("Saved predictions differ from scores and cutoff")
    return p


def declared_files(folder):
    """Enumerate a trusted local manifest without allowing path traversal."""
    folder = folder.resolve()
    meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    result = [folder / "metadata.json"]
    for name in meta["files"]:
        target = (folder / name).resolve()
        if not target.is_relative_to(folder):
            raise ValueError("Artifact file outside its bundle")
        result.append(target)
    return result


def input_inventory(settings):
    models, prepared, report = (settings[n] for n in ("models", "prepared", "report"))
    files = declared_files(models) + declared_files(report)
    selection_file = models / "selection.json"
    if selection_file.exists():
        from integritree.ml.staged_selection import resolve_reference

        selection = json.loads(selection_file.read_text(encoding="utf-8"))
        files += declared_files(resolve_reference(selection["model"], models))
    files += [
        prepared / name
        for name in (
            "metadata.json",
            "configuration.json",
            "preprocessing.json",
            "split_manifest.parquet",
            "source.parquet",
            "validation_features.parquet",
            "train_features.parquet",
        )
    ]
    # No test feature file is opened, including for hashing.
    return {str(p): file_sha256(p) for p in sorted(set(files))}


def verify_unchanged(before):
    for path, expected in before.items():
        if file_sha256(Path(path)) != expected:
            raise ValueError(f"Input artifact changed during investigation: {path}")


def load_inputs(settings, progress):
    models, prepared, report = (settings[n] for n in ("models", "prepared", "report"))
    # Reject test evidence before loading models or feature files.
    meta, predictions = load_report(report)
    if meta.get("split") != "validation" or meta.get("official_test") is not False:
        raise ValueError("Investigation accepts validation reports only")
    progress("Verifying saved model selection and loading fitted pair")
    bundle = load_bundle(models)
    verify_links(meta, bundle, models, prepared)
    prep_meta = verify_prepared(bundle, prepared)
    progress("Checking validation identities, features, labels, and saved metrics")
    features, labels, rows = load_prepared_split(prepared, "validation")
    p = align_predictions(
        predictions,
        rows,
        labels,
        bundle.config.dataset.sha256,
        bundle.metadata["run_id"],
        bundle.config.scoring.threshold,
    )
    policy = EvaluationConfig.model_validate(meta["policy"])
    saved_metrics = json.loads((report / "metrics.json").read_text(encoding="utf-8"))
    if saved_metrics.get("split") != "validation":
        raise ValueError("Saved metrics are not validation results")
    for model in ("rf", "rf_smote"):
        actual = metrics(
            p.actual_label, p[f"{model}_risk_score"], meta["threshold"], policy
        )
        if actual != saved_metrics["models"][model]:
            raise ValueError(f"Saved metrics do not match predictions: {model}")
    source = prepared / "source.parquet"
    if file_sha256(source) != prep_meta["files"]["source.parquet"]:
        raise ValueError("Source fingerprint mismatch")
    progress("Joining validation-only source attributes in bounded batches")
    wanted = pd.Index(rows)
    pieces = []
    for batch in pq.ParquetFile(source).iter_batches(
        batch_size=100000, columns=["source_row_number", "step", "type", "amount"]
    ):
        part = batch.to_pandas()
        pieces.append(part.loc[part.source_row_number.isin(wanted)])
    source_rows = pd.concat(pieces, ignore_index=True)
    if len(source_rows) != len(p) or not source_rows.source_row_number.is_unique:
        raise ValueError("Missing or duplicate source attributes")
    p = p.merge(source_rows, on="source_row_number", how="left", validate="one_to_one")
    # Analyze model-visible missing amounts using the fitted training median.
    p["amount_was_missing"] = p.amount.isna()
    p["amount"] = p.amount.fillna(bundle.preprocessor.state.amount_median)
    p["type"] = p.type.fillna("UNKNOWN")
    if p.step.isna().any() or not np.isfinite(p.amount).all() or (p.amount < 0).any():
        raise ValueError("Invalid source attributes")
    features = features.copy()
    features.index = pd.Index(rows, name="source_row_number")
    for name in ("is_zero_amount", "is_merchant_origin", "is_merchant_dest"):
        p[name] = features.loc[p.source_row_number, name].to_numpy()
    if list(features.columns) != FEATURE_COLUMNS:
        raise ValueError("Unexpected feature order")
    return bundle, p, features, policy, meta
