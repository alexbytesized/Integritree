"""Shared exact selection arithmetic and candidate memory management."""

from fractions import Fraction
import gc

import pyarrow as pa

from integritree.ml.evaluation import binary_labels, score_vector


def exact_mean_f1(y, rf, smote, threshold):
    y = binary_labels(y)
    if set(y.tolist()) != {0, 1}:
        raise ValueError("Selection requires both ground-truth classes")
    values = []
    for scores in (rf, smote):
        pred = score_vector(scores, len(y)) >= threshold
        tp = int(((y == 1) & pred).sum())
        fp = int(((y == 0) & pred).sum())
        fn = int(((y == 1) & ~pred).sum())
        values.append(Fraction(2 * tp, 2 * tp + fp + fn))
    return sum(values) / 2


def release_candidate_memory():
    """Release unused report/Arrow allocations without bypassing RAM checks."""
    gc.collect()
    pa.default_memory_pool().release_unused()
