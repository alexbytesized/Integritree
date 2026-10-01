"""Explicit metric and statistical policies for paired binary fraud predictions."""

import math
import numpy as np
from scipy.stats import binomtest, chi2
from sklearn.metrics import average_precision_score
from integritree.config import EvaluationConfig
from integritree.ml.inference import classify

METRICS = ("precision", "recall", "f1", "mcc", "pr_auc", "accuracy")


def validate_policy(policy: EvaluationConfig):
    expected = {
        "pr_auc_method": "average_precision",
        "mcnemar_method": "continuity_corrected",
        "discordance_edge_policy": "exact_supplement_1_to_24_zero_p1",
        "undefined_metric_policy": "null_with_reason",
        "absent_positive_pr_policy": "null_with_reason",
    }
    for name, value in expected.items():
        if getattr(policy, name) != value:
            raise ValueError(f"Unsupported or unresolved evaluation.{name}")


def binary_labels(values):
    y = np.asarray(values)
    if y.ndim != 1 or not len(y) or not np.isin(y, [0, 1]).all():
        raise ValueError("Labels must be nonempty binary 0/1 values")
    return y.astype("uint8")


def score_vector(values, rows):
    scores = np.asarray(values, dtype="float64")
    if (
        scores.shape != (rows,)
        or not np.isfinite(scores).all()
        or ((scores < 0) | (scores > 1)).any()
    ):
        raise ValueError("Scores must be aligned finite probabilities")
    return scores


def measured(value, reason=None):
    return {
        "value": None if value is None else float(value),
        "status": "unavailable" if value is None else "available",
        "reason": reason,
    }


def metrics(labels, scores, threshold, policy):
    validate_policy(policy)
    y = binary_labels(labels)
    scores = score_vector(scores, len(y))
    pred = classify(scores, threshold)
    tp = int(((y == 1) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())

    def ratio(n, d, reason):
        return measured(n / d) if d else measured(None, reason)

    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    values = {
        "precision": ratio(tp, tp + fp, "no_predicted_positives"),
        "recall": ratio(tp, tp + fn, "no_actual_positives"),
        # Calculate directly from counts even when precision is undefined.
        "f1": ratio(2 * tp, 2 * tp + fp + fn, "no_actual_or_predicted_positives"),
        "mcc": ratio(tp * tn - fp * fn, denominator, "zero_mcc_denominator"),
        "pr_auc": measured(average_precision_score(y, scores))
        if tp + fn
        else measured(None, "no_actual_positives"),
        "accuracy": measured((tp + tn) / len(y)),
    }
    return {
        "rows": len(y),
        "actual_fraud": tp + fn,
        "positive_class": 1,
        "threshold": threshold,
        "tie_policy": "fraud",
        "pr_auc_method": "average_precision",
        "pr_auc_label": "PR-AUC (Average Precision)",
        "metrics": values,
        "confusion_matrix": {
            "axes": {"rows": "Actual", "columns": "Predicted"},
            "labels": [0, 1],
            "matrix": [[tn, fp], [fn, tp]],
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        },
    }


def comparison(benchmark, experimental, name, method):
    if method not in ("signed_over_mean", "absolute_over_mean"):
        raise ValueError("Unknown descriptive comparison method")
    if benchmark is None or experimental is None:
        return measured(None, "undefined_input") | {
            "method": method,
            "units": "percent",
        }
    delta = experimental - benchmark
    if name == "mcc" and min(benchmark, experimental) < 0:
        return measured(delta) | {
            "method": "signed_coefficient_difference",
            "units": "coefficient",
        }
    mean = (benchmark + experimental) / 2
    if mean <= 0:
        return measured(None, "zero_denominator") | {
            "method": method,
            "units": "percent",
        }
    value = 100 * (abs(delta) if method == "absolute_over_mean" else delta) / mean
    return measured(value) | {"method": method, "units": "percent"}


def mcnemar(labels, rf_pred, smote_pred, policy):
    validate_policy(policy)
    y = binary_labels(labels)
    a_pred, b_pred = binary_labels(rf_pred), binary_labels(smote_pred)
    if len(y) != len(a_pred) or len(y) != len(b_pred):
        raise ValueError("Paired predictions must be aligned")
    rf_ok, smote_ok = a_pred == y, b_pred == y
    a = int((rf_ok & smote_ok).sum())
    b = int((rf_ok & ~smote_ok).sum())
    c = int((~rf_ok & smote_ok).sum())
    d = int((~rf_ok & ~smote_ok).sum())
    discordant = b + c
    statistic = ((abs(b - c) - 1) ** 2 / discordant) if discordant else None
    p = float(chi2.sf(statistic, 1)) if discordant else 1.0
    supplement = None
    if 1 <= discordant <= 24:
        supplement = {
            "method": "exact_binomial_two_sided",
            "p_value": float(binomtest(b, discordant, 0.5).pvalue),
            "reason": "1_to_24_discordant_pairs",
            "role": "supplementary",
        }
    return {
        "method": "continuity_corrected",
        "role": "primary",
        "alpha": policy.alpha,
        "table": {
            "rows": "RF correctness",
            "columns": "RF-SMOTE correctness",
            "labels": ["correct", "incorrect"],
            "matrix": [[a, b], [c, d]],
            "both_correct": a,
            "rf_only_correct": b,
            "smote_only_correct": c,
            "both_incorrect": d,
        },
        "discordant_pairs": discordant,
        "statistic": statistic,
        "p_value": p,
        "status": "No discordant pairs" if not discordant else "computed",
        "p_value_convention": "1_when_no_discordant_pairs" if not discordant else None,
        "approximation_caution": 1 <= discordant <= 24,
        "decision": "reject_null" if p < policy.alpha else "fail_to_reject_null",
        "lower_observed_error_model": "rf_smote" if c > b else "rf" if b > c else None,
        "scope": "paired classification error rates; not significance of individual metrics",
        "supplementary": supplement,
    }


def evaluate_pair(labels, rf_scores, smote_scores, threshold, policy):
    y = binary_labels(labels)
    rf_scores, smote_scores = (
        score_vector(rf_scores, len(y)),
        score_vector(smote_scores, len(y)),
    )
    models = {
        "rf": metrics(y, rf_scores, threshold, policy),
        "rf_smote": metrics(y, smote_scores, threshold, policy),
    }
    differences = {
        name: comparison(
            models["rf"]["metrics"][name]["value"],
            models["rf_smote"]["metrics"][name]["value"],
            name,
            policy.percentage_difference,
        )
        for name in METRICS
    }
    return {
        "models": models,
        "descriptive_comparisons": differences,
        "statistical_test": mcnemar(
            y, classify(rf_scores, threshold), classify(smote_scores, threshold), policy
        ),
    }
