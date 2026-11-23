"""Fixed protocol, source validation, local artifacts, and matched comparisons."""

from importlib.metadata import version
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import yaml

from integritree.ml.artifacts import PACKAGES
from integritree.ml.data import file_sha256, write_json
from integritree.ml.features import FEATURE_COLUMNS
from integritree.ml.inference import fraud_scores, feature_matrix
from integritree.ml.threshold_search import percent_grid_thresholds
from integritree.ml.training import forest_parameters
from investigation.diagnostics.analysis import add_outcomes, metric_row

ARTIFACT_KIND = "isolated_smote_ratio"
ARTIFACT_FILES = {
    "rf_smote.joblib",
    "configuration.json",
    "preprocessing.json",
    "training_audit.json",
    "reload_verification.json",
    "synthetic_features.parquet",
}


def read_config(path):
    path = Path(path).resolve()
    settings = yaml.safe_load(path.read_text(encoding="utf-8"))
    keys = {
        "schema_version",
        "models",
        "prepared",
        "report",
        "ratio_denominators",
        "sample_per_group",
        "seed",
        "minimum_denominator",
    }
    if (
        not isinstance(settings, dict)
        or set(settings) != keys
        or settings["schema_version"] != 1
    ):
        raise ValueError("Unsupported ratio experiment configuration")
    # This first automation is the three comparisons authorized by the researcher.
    if settings["ratio_denominators"] != [200, 300, 500] or any(
        type(x) is not int for x in settings["ratio_denominators"]
    ):
        raise ValueError("This protocol requires ratios 1:200, 1:300, 1:500 in order")
    if type(settings["seed"]) is not int or settings["seed"] != 42:
        raise ValueError("This protocol requires seed 42; no seed search")
    for key in ("sample_per_group", "minimum_denominator"):
        if type(settings[key]) is not int or settings[key] < 1:
            raise ValueError(f"Invalid {key}")
    if settings["sample_per_group"] > 5:
        raise ValueError("At most five sampled cases per outcome/type group")
    for key in ("models", "prepared", "report"):
        if not isinstance(settings[key], str) or not settings[key].strip():
            raise ValueError(f"Missing {key} path")
        settings[key] = (path.parent / settings[key]).resolve()
    return settings


def verify_reference(bundle):
    config = bundle.config
    if (
        config.smote.method != "smote_encoded"
        or config.smote.sampling_ratio != 0.01
        or config.smote.k_neighbors != 5
        or config.seeds.smote != 42
        or config.seeds.model != 42
        or config.scoring.threshold != 0.43
        or config.shap.background_size != 200
    ):
        raise ValueError("Reference differs from the approved 1:100 / 43% protocol")
    parameters = forest_parameters(config, jobs=1)
    if any(
        parameters[n] != v
        for n, v in {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_leaf": 1,
        }.items()
    ):
        raise ValueError("Reference forest differs from the approved fixed forest")
    for model in bundle.models.values():
        if any(model.get_params()[k] != v for k, v in parameters.items()):
            raise ValueError("Reference forest parameters do not match")
    return parameters


def verify_training_ids(train_ids, validation_ids):
    train, validation = pd.Index(train_ids), pd.Index(validation_ids)
    if (
        train.hasnans
        or validation.hasnans
        or not train.is_unique
        or not validation.is_unique
    ):
        raise ValueError("Missing or duplicate training/validation identities")
    if len(train.intersection(validation)):
        raise ValueError("Training and validation membership overlap")


def predict(model, features):
    parts = []
    for start in range(0, len(features), 25000):
        batch = (
            features.iloc[start : start + 25000]
            if isinstance(features, pd.DataFrame)
            else features[start : start + 25000]
        )
        values = (
            feature_matrix(batch)
            if isinstance(batch, pd.DataFrame)
            else np.asarray(batch, dtype="float64")
        )
        parts.append(fraud_scores(model, values))
    return np.concatenate(parts)


def make_pair(reference, scores, threshold):
    """Only copied validation records are thresholded; score ties predict fraud."""
    scores = np.asarray(scores)
    if (
        len(scores) != len(reference)
        or scores.ndim != 1
        or not np.isfinite(scores).all()
        or ((scores < 0) | (scores > 1)).any()
    ):
        raise ValueError("Invalid candidate scores")
    frame = reference.copy()
    frame["reference_100_risk_score"] = frame.rf_smote_risk_score
    frame["rf_smote_risk_score"] = scores
    frame["threshold"] = threshold
    for name in ("rf", "rf_smote", "reference_100"):
        frame[f"{name}_predicted_label"] = (
            frame[f"{name}_risk_score"] >= threshold
        ).astype("uint8")
    return add_outcomes(frame)


def overview(frame, policy, minimum, denominator, mode):
    cutoff = float(frame.threshold.iloc[0])
    rows = []
    for name, label in (
        ("rf", "RF"),
        ("rf_smote", f"RF-SMOTE 1:{denominator}"),
        ("reference_100", "RF-SMOTE 1:100 reference"),
    ):
        item = metric_row(frame, name, cutoff, policy, minimum)
        item.update(
            model=label, ratio_denominator=denominator, mode=mode, threshold=cutoff
        )
        rows.append(item)
    return pd.DataFrame(rows)


def choose_cutoff(reference, scores):
    return percent_grid_thresholds(
        reference.actual_label, reference.rf_risk_score, scores
    )


def save_model_metadata(folder, reference, parameters, config, audit, verification):
    write_json(folder / "configuration.json", config.model_dump())
    write_json(folder / "training_audit.json", audit)
    write_json(folder / "reload_verification.json", verification)
    metadata = {
        "schema_version": 1,
        "kind": ARTIFACT_KIND,
        "status": "complete",
        "run_id": folder.parent.name,
        "feature_order": FEATURE_COLUMNS,
        "prepared_metadata_sha256": reference.metadata["prepared_metadata_sha256"],
        "rf_parameters": parameters,
        "test_used": False,
        "new_fits": 1,
        "package_versions": {p: version(p) for p in PACKAGES},
        "files": {name: file_sha256(folder / name) for name in sorted(ARTIFACT_FILES)},
    }
    write_json(folder / "metadata.json", metadata)
    return metadata


def load_model(folder, prepared_hash=None):
    """Local-only loader. This format cannot be loaded as an application bundle."""
    meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    if (
        meta.get("kind") != ARTIFACT_KIND
        or meta.get("status") != "complete"
        or meta.get("schema_version") != 1
        or meta.get("feature_order") != FEATURE_COLUMNS
        or set(meta.get("files", {})) != ARTIFACT_FILES
    ):
        raise ValueError("Incomplete or incompatible isolated artifact")
    if (
        prepared_hash is not None
        and meta.get("prepared_metadata_sha256") != prepared_hash
    ):
        raise ValueError("Prepared provenance mismatch")
    for name, expected in meta["files"].items():
        if file_sha256(folder / name) != expected:
            raise ValueError(f"Artifact fingerprint mismatch: {name}")
    for name in PACKAGES:
        if meta["package_versions"].get(name) != version(name):
            raise ValueError(f"Artifact dependency mismatch: {name}")
    model = joblib.load(folder / "rf_smote.joblib")
    if (
        not isinstance(model, RandomForestClassifier)
        or model.n_features_in_ != len(FEATURE_COLUMNS)
        or not np.array_equal(model.classes_, [0, 1])
        or any(model.get_params()[k] != v for k, v in meta["rf_parameters"].items())
    ):
        raise ValueError("Saved model does not match metadata")
    return model
