"""Exact shared-F1 comparison on the 1% through 100% threshold grid."""

from fractions import Fraction

import numpy as np
import pandas as pd

from integritree.ml.evaluation import binary_labels, score_vector


def percent_grid_thresholds(labels, rf_scores, smote_scores):
    """O(N log N) count sweep; no N-by-threshold matrix or repeated predictions."""
    y = binary_labels(labels)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Threshold selection requires both classes")
    scores = [score_vector(s, len(y)) for s in (rf_scores, smote_scores)]
    percentages = np.arange(1, 101)
    thresholds = percentages / 100
    data = {"threshold_percent": percentages, "threshold": thresholds}
    positives = int(y.sum())
    negatives = len(y) - positives
    counts = []
    for name, values in zip(("rf", "rf_smote"), scores):
        order = np.argsort(values, kind="stable")
        # Left insertion includes every score exactly equal to the threshold.
        excluded = np.searchsorted(values[order], thresholds, side="left")
        prefix = np.concatenate([[0], np.cumsum(y[order], dtype=np.int64)])
        fn = prefix[excluded]
        tp = positives - fn
        tn = excluded - fn
        fp = negatives - tn
        denominator = 2 * tp + fp + fn
        counts.append((tp, denominator))
        for key, array in (("tp", tp), ("fp", fp), ("tn", tn), ("fn", fn)):
            data[f"{name}_{key}"] = array
        data[f"{name}_precision"] = np.divide(
            tp, tp + fp, out=np.full(len(tp), np.nan), where=tp + fp != 0
        )
        data[f"{name}_recall"] = tp / positives
        data[f"{name}_f1"] = 2 * tp / denominator
        mcc_den = np.sqrt((tp + fp).astype(float) * (tp + fn) * (tn + fp) * (tn + fn))
        data[f"{name}_mcc"] = np.divide(
            tp.astype(float) * tn - fp.astype(float) * fn,
            mcc_den,
            out=np.full(len(tp), np.nan),
            where=mcc_den != 0,
        )
        data[f"{name}_accuracy"] = (tp + tn) / len(y)
    data["mean_f1"] = (data["rf_f1"] + data["rf_smote_f1"]) / 2
    best_index, best_num, best_den = 0, -1, 1
    for i, (a, b, c, d) in enumerate(
        zip(counts[0][0], counts[0][1], counts[1][0], counts[1][1])
    ):
        # Python integers prevent overflow in rational cross-products.
        a, b, c, d = int(a), int(b), int(c), int(d)
        numerator, denominator = a * d + c * b, b * d
        comparison = numerator * best_den - best_num * denominator
        if comparison == 0:
            # Resolve exact F1 ties by proximity to 50%, then the higher cutoff.
            candidate = int(percentages[i])
            previous = int(percentages[best_index])
            better_tie = (-abs(candidate - 50), candidate) > (
                -abs(previous - 50),
                previous,
            )
        else:
            better_tie = False
        if comparison > 0 or better_tie:
            best_index, best_num, best_den = i, numerator, denominator
    best = {
        "threshold": float(thresholds[best_index]),
        "mean_f1_fraction": str(Fraction(best_num, best_den)),
        "candidate_count": len(thresholds),
    }
    return best, pd.DataFrame(data)
