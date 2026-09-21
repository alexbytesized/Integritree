"""Integrity-checked local model bundles. Load only trusted local joblib files."""
from dataclasses import dataclass
from importlib.metadata import version
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from integritree.config import ExperimentConfig
from integritree.ml.data import file_sha256
from integritree.ml.features import FEATURE_COLUMNS
from integritree.ml.preprocessing import FittedPreprocessor

PACKAGES = ("integritree-backend", "numpy", "pandas", "pyarrow", "scikit-learn",
            "imbalanced-learn", "joblib")
REQUIRED_FILES = {"rf.joblib", "rf_smote.joblib", "preprocessing.json", "configuration.json",
                  "prepared_metadata.json", "training_audit.json", "reload_verification.json"}


@dataclass
class ModelBundle:
    models: dict
    preprocessor: FittedPreprocessor
    config: ExperimentConfig
    metadata: dict


def load_bundle(path: Path) -> ModelBundle:
    """Hashes detect corruption, not a maliciously replaced manifest/pickle."""
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("status") != "complete" or metadata.get("schema_version") != 1:
        raise ValueError("Model bundle is incomplete or unsupported")
    if metadata.get("feature_order") != FEATURE_COLUMNS:
        raise ValueError("Unsupported artifact feature order")
    if set(metadata.get("files", {})) != REQUIRED_FILES:
        raise ValueError("Model bundle file manifest is incomplete or unsupported")
    for name, expected in metadata["files"].items():
        if file_sha256(path / name) != expected:
            raise ValueError(f"Model bundle fingerprint mismatch: {name}")
    for name in PACKAGES:
        if metadata.get("package_versions", {}).get(name) != version(name):
            raise ValueError(f"Artifact dependency version mismatch: {name}")
    config = ExperimentConfig.model_validate_json((path / "configuration.json").read_text())
    config.require_ready("train")
    if config.scoring.probability_method != "mean_tree_probability" or config.scoring.tie_policy != "fraud":
        raise ValueError("Unsupported saved scoring policy")
    preprocessor = FittedPreprocessor.load(path / "preprocessing.json")
    if config.preprocessing.features != FEATURE_COLUMNS or config.dataset.model_dump() != metadata.get("dataset"):
        raise ValueError("Saved configuration does not match artifact provenance/schema")
    models = {name: joblib.load(path / f"{name}.joblib") for name in ("rf", "rf_smote")}
    from integritree.ml.training import forest_parameters, validate_training_config
    validate_training_config(config)
    expected_parameters = forest_parameters(config, metadata["rf_parameters"]["n_jobs"])
    for model in models.values():
        if not isinstance(model, RandomForestClassifier) or model.n_features_in_ != len(FEATURE_COLUMNS):
            raise ValueError("Unsupported saved model")
        if not np.array_equal(model.classes_, [0, 1]):
            raise ValueError("Saved model classes must be 0 and 1")
        if any(model.get_params()[key] != value for key, value in expected_parameters.items()):
            raise ValueError("Saved model settings differ from the configuration")
    if models["rf"].get_params() != models["rf_smote"].get_params():
        raise ValueError("RF model settings do not match")
    return ModelBundle(models, preprocessor, config, metadata)
