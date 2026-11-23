from fractions import Fraction

import numpy as np
import pandas as pd

from diagnostics.analysis import (
    AMOUNT_LABELS,
    OUTCOMES,
    add_outcomes,
    attainable_points,
    budget_comparisons,
    findings,
    grid_diagnostics,
    metric_row,
    outcome_counts,
    sample_cases,
    subgroup_metrics,
)


def test_all_eight_outcomes_and_amount_time_boundaries(frame):
    result = add_outcomes(frame)
    assert result.outcome.tolist() == list(OUTCOMES)
    assert outcome_counts(result).rows.tolist() == [1] * 8
    assert result.amount_band.tolist() == [
        AMOUNT_LABELS[i] for i in [0, 0, 1, 1, 2, 2, 3, 4]
    ]
    assert result.hour.iloc[:5].tolist() == [0, 23, 0, 23, 0]
    assert result.day.iloc[:5].tolist() == [0, 0, 1, 6, 0]


def test_absent_outcomes_are_reported_as_zero(frame):
    result = outcome_counts(add_outcomes(frame.iloc[[0, 7]]))
    assert result.rows.tolist() == [1, 0, 0, 0, 0, 0, 0, 1]


def test_hand_counted_metrics_and_denominators(frame, config):
    row = metric_row(frame, "rf", 0.43, config.evaluation, 5)
    assert [row[k] for k in ["tp", "fp", "tn", "fn"]] == [2] * 4
    assert row["precision"] == row["recall"] == row["f1"] == 0.5
    assert row["mcc"] == 0
    assert row["false_alarms_per_10000"] == 5000
    assert row["recall_denominator"] == row["precision_denominator"] == 4
    assert row["mcc_denominator"] == 4
    assert row["f1_denominator"] == 8
    assert row["precision_sparse"] and not row["f1_sparse"]


def test_undefined_metrics_and_sparse_groups(frame, config):
    row = metric_row(frame.iloc[[0]], "rf", 0.43, config.evaluation, 20)
    for name in ("precision", "recall", "mcc", "pr_auc", "f1"):
        assert row[name] is None
        assert row[f"{name}_reason"]
        assert row[f"{name}_sparse"]
    groups = subgroup_metrics(add_outcomes(frame), 0.43, config.evaluation, 20)
    notes = findings(add_outcomes(frame), groups, 20)
    assert not any("highest false-alarm rate" in note for note in notes)


def test_grid_matches_direct_predictions_and_tie_rule(frame):
    grid, best, shared = grid_diagnostics(frame)
    assert len(grid) == 100
    # This deliberately uninformative fixture favors predicting every row positive.
    assert shared["threshold"] == 0.1
    assert best.threshold.tolist() == [0.2, 0.1]
    for model in ("rf", "rf_smote"):
        ranked = []
        for _, row in grid.iterrows():
            pred = frame[f"{model}_risk_score"] >= row.threshold
            y = frame.actual_label == 1
            tp, fp, fn = (
                int((pred & y).sum()),
                int((pred & ~y).sum()),
                int((~pred & y).sum()),
            )
            assert (row[f"{model}_tp"], row[f"{model}_fp"], row[f"{model}_fn"]) == (
                tp,
                fp,
                fn,
            )
            percent = int(row.threshold_percent)
            ranked.append(
                (Fraction(2 * tp, 2 * tp + fp + fn), -abs(percent - 50), percent)
            )
        assert (
            best.loc[best.model == model, "threshold"].iloc[0] == max(ranked)[2] / 100
        )


def test_attainable_points_do_not_split_ties_and_include_no_alerts():
    points = attainable_points([0, 1, 0, 1], [1.0, 0.8, 0.8, 0.3])
    assert points.tp.tolist() == [0, 0, 1, 2]
    assert points.fp.tolist() == [0, 1, 2, 2]
    assert points.iloc[0].no_alerts and pd.isna(points.iloc[0].threshold)
    assert not ((points.tp == 1) & (points.fp == 1)).any()


def test_grid_equal_f1_prefers_half(frame):
    perfect = frame.copy()
    for model in ("rf", "rf_smote"):
        perfect[f"{model}_risk_score"] = np.where(perfect.actual_label, 0.8, 0.2)
    _, best, shared = grid_diagnostics(perfect)
    assert shared["threshold"] == 0.5
    assert best.threshold.tolist() == [0.5, 0.5]


def test_budget_does_not_exceed_observed_false_alarms(frame):
    result = budget_comparisons(frame)
    assert (result.fp <= result.false_alarm_budget).all()
    assert len(result) == 4
    assert np.allclose(result.recall, 0.5)


def test_sampling_is_order_independent_and_preserves_small_cells(frame):
    repeated = pd.concat(
        [
            frame.assign(
                source_row_number=frame.source_row_number + 8 * i,
                transaction_id=[f"sample:{8 * i + j}" for j in range(8)],
            )
            for i in range(12)
        ],
        ignore_index=True,
    )
    enriched = add_outcomes(repeated)
    selected, coverage = sample_cases(enriched, 5, 42)
    reordered, _ = sample_cases(enriched.sample(frac=1, random_state=6), 5, 42)
    assert selected.source_row_number.tolist() == reordered.source_row_number.tolist()
    assert len(selected) == 40 and coverage.sampled.eq(5).all()
    small, coverage = sample_cases(add_outcomes(frame), 5, 42)
    assert len(small) == 8 and coverage.sampled.eq(1).all()
