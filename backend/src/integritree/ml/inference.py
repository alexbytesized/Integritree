"""Shared paired inference. Labels and identity columns never enter the models."""
import numpy as np
import pandas as pd
from integritree.ml.features import FEATURE_COLUMNS


def feature_matrix(features: pd.DataFrame) -> np.ndarray:
    if list(features.columns) != FEATURE_COLUMNS:
        raise ValueError("Feature columns/order must exactly match the saved schema")
    values = features.to_numpy(dtype="float64", copy=True)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Features must be nonempty and finite")
    return values


def fraud_scores(model, values: np.ndarray) -> np.ndarray:
    classes = np.asarray(model.classes_)
    if classes.ndim != 1 or set(classes.tolist()) != {0, 1}:
        raise ValueError("Model must have binary classes 0 and 1")
    scores = model.predict_proba(values)[:, int(np.flatnonzero(classes == 1)[0])]
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Invalid model probabilities")
    return scores


def classify(scores, threshold: float) -> np.ndarray:
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be finite and within [0, 1]")
    return (np.asarray(scores) >= threshold).astype("uint8")


def predict_features(models: dict, features: pd.DataFrame, transaction_ids,
                     threshold: float, run_id: str) -> pd.DataFrame:
    values = feature_matrix(features)
    ids = list(transaction_ids)
    if len(ids) != len(values) or any(not isinstance(i, str) or not i.strip() for i in ids):
        raise ValueError("A nonempty string transaction ID is required for every row")
    if len(set(ids)) != len(ids):
        raise ValueError("Transaction IDs must be unique within a batch")
    if set(models) != {"rf", "rf_smote"}:
        raise ValueError("Both RF and RF-SMOTE are required")
    result = pd.DataFrame({"transaction_id": ids, "run_id": run_id})
    for name in ("rf", "rf_smote"):
        scores = fraud_scores(models[name], values)
        result[f"{name}_risk_score"] = scores
        result[f"{name}_predicted_label"] = classify(scores, threshold)
    return result


def predict_records(bundle, inputs: pd.DataFrame, transaction_ids) -> pd.DataFrame:
    """Apply the saved preprocessor without refitting; supports one or many rows."""
    return predict_features(bundle.models, bundle.preprocessor.transform(inputs),
                            transaction_ids, bundle.config.scoring.threshold,
                            bundle.metadata["run_id"])
