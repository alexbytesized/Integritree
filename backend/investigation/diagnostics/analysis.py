"""Pure diagnostic calculations with explicit counts and support."""

from fractions import Fraction

import numpy as np
import pandas as pd

from integritree.ml.evaluation import metrics
from integritree.ml.threshold_search import percent_grid_thresholds

MODELS = ("rf", "rf_smote")
OUTCOMES = (
    "both_correct_legitimate",
    "smote_only_false_alarm",
    "rf_only_false_alarm",
    "both_false_alarm",
    "both_missed_fraud",
    "smote_only_detected_fraud",
    "rf_only_detected_fraud",
    "both_detected_fraud",
)
AMOUNT_LABELS = (
    "<1,000",
    "1,000–<10,000",
    "10,000–<100,000",
    "100,000–<1,000,000",
    "≥1,000,000",
)
GROUPINGS = (
    ("type",),
    ("amount_band",),
    ("hour",),
    ("day",),
    ("is_zero_amount",),
    ("is_merchant_origin",),
    ("is_merchant_dest",),
    ("type", "amount_band"),
    ("type", "hour"),
)


def add_outcomes(frame):
    frame = frame.copy()
    code = (
        4 * frame.actual_label.astype(int)
        + 2 * frame.rf_predicted_label.astype(int)
        + frame.rf_smote_predicted_label.astype(int)
    )
    frame["outcome"] = np.asarray(OUTCOMES)[code]
    frame["amount_band"] = pd.cut(
        frame.amount,
        [-np.inf, 1000, 10000, 100000, 1000000, np.inf],
        labels=AMOUNT_LABELS,
        right=False,
    )
    frame["hour"] = (frame.step.astype(int) - 1) % 24
    frame["day"] = ((frame.step.astype(int) - 1) // 24) % 7
    return frame


def outcome_counts(frame):
    return (
        frame.outcome.value_counts()
        .reindex(OUTCOMES, fill_value=0)
        .rename_axis("outcome")
        .reset_index(name="rows")
    )


def metric_row(frame, model, threshold, policy, minimum):
    measured = metrics(
        frame.actual_label, frame[f"{model}_risk_score"], threshold, policy
    )
    counts = measured["confusion_matrix"]
    tp, fp, tn, fn = (counts[n] for n in ("tp", "fp", "tn", "fn"))
    supports = {
        "precision": tp + fp,
        "recall": tp + fn,
        "f1": 2 * tp + fp + fn,
        # MCC is supported by four marginals; expose the smallest one.
        "mcc": min(tp + fp, tp + fn, tn + fp, tn + fn),
        "pr_auc": tp + fn,
        "accuracy": len(frame),
        "false_alarms_per_10000": tn + fp,
    }
    result = dict(
        model=model,
        rows=len(frame),
        legitimate=tn + fp,
        fraud=tp + fn,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
    )
    for name, item in measured["metrics"].items():
        result[name] = item["value"]
        result[f"{name}_reason"] = item["reason"]
    result["false_alarms_per_10000"] = 10000 * fp / (tn + fp) if tn + fp else None
    result["false_alarms_per_10000_reason"] = None if tn + fp else "no_actual_negatives"
    for name, support in supports.items():
        result[f"{name}_denominator"] = support
        result[f"{name}_sparse"] = support < minimum
    return result


def subgroup_metrics(frame, threshold, policy, minimum):
    rows = []
    for columns in GROUPINGS:
        for keys, group in frame.groupby(list(columns), observed=True, sort=True):
            if not isinstance(keys, tuple):
                keys = (keys,)
            identity = {
                "grouping": " × ".join(columns),
                "group": " | ".join(map(str, keys)),
            }
            for model in MODELS:
                rows.append(
                    identity | metric_row(group, model, threshold, policy, minimum)
                )
    return pd.DataFrame(rows)


def grid_diagnostics(frame):
    shared, table = percent_grid_thresholds(
        frame.actual_label, frame.rf_risk_score, frame.rf_smote_risk_score
    )
    best = []
    for model in MODELS:
        # Same exact F1 comparison and tie order as the shared production sweep.
        def rank(row):
            tp, fp, fn = (int(row[f"{model}_{n}"]) for n in ("tp", "fp", "fn"))
            percent = int(row.threshold_percent)
            return Fraction(2 * tp, 2 * tp + fp + fn), -abs(percent - 50), percent

        winner = max((row for _, row in table.iterrows()), key=rank)
        best.append(
            {
                "model": model,
                "threshold": winner.threshold,
                **{
                    n: winner[f"{model}_{n}"]
                    for n in (
                        "tp",
                        "fp",
                        "tn",
                        "fn",
                        "precision",
                        "recall",
                        "f1",
                        "mcc",
                    )
                },
            }
        )
    return table, pd.DataFrame(best), shared


def attainable_points(labels, scores):
    """Every attainable score-tied operating point, including no alerts."""
    labels, scores = np.asarray(labels), np.asarray(scores)
    order = np.argsort(-scores, kind="stable")
    ordered, truth = scores[order], labels[order]
    ends = np.r_[np.flatnonzero(ordered[:-1] != ordered[1:]), len(scores) - 1]
    tp = np.cumsum(truth, dtype=np.int64)[ends]
    table = pd.DataFrame(
        {"threshold": ordered[ends], "tp": tp, "fp": ends + 1 - tp, "no_alerts": False}
    )
    # NaN deliberately represents a no-alert rule, not a deployable probability cutoff.
    return pd.concat(
        [
            pd.DataFrame([{"threshold": np.nan, "tp": 0, "fp": 0, "no_alerts": True}]),
            table,
        ],
        ignore_index=True,
    )


def budget_comparisons(frame):
    budgets = {
        model: int(
            ((frame.actual_label == 0) & (frame[f"{model}_predicted_label"] == 1)).sum()
        )
        for model in MODELS
    }
    rows = []
    for model in MODELS:
        points = attainable_points(frame.actual_label, frame[f"{model}_risk_score"])
        for reference, budget in budgets.items():
            eligible = points[points.fp <= budget]
            # Maximize TP, then minimize FP; stable ordering prefers higher thresholds.
            chosen = eligible.sort_values(
                ["tp", "fp"], ascending=[False, True], kind="stable"
            ).iloc[0]
            rows.append(
                {
                    "budget_source": reference,
                    "false_alarm_budget": budget,
                    "model": model,
                    **chosen.to_dict(),
                    "recall": float(chosen.tp / frame.actual_label.sum()),
                }
            )
    return pd.DataFrame(rows)


def score_distributions(frame):
    rows = []
    for outcome, group in frame.groupby("outcome", sort=True):
        for model in MODELS:
            score = group[f"{model}_risk_score"]
            rows.append(
                {
                    "outcome": outcome,
                    "model": model,
                    "rows": len(group),
                    **{
                        f"q{int(q * 100):02}": float(score.quantile(q))
                        for q in (0, 0.1, 0.25, 0.5, 0.75, 0.9, 1)
                    },
                }
            )
    return pd.DataFrame(rows)


def sample_cases(frame, count, seed):
    # Each cell gets an independent stable RNG: absent cells cannot perturb others.
    selected, coverage = [], []
    for (outcome, kind), group in frame.groupby(["outcome", "type"], sort=True):
        group = group.sort_values("source_row_number")
        chosen = np.sort(
            np.random.default_rng(seed).choice(
                len(group), min(count, len(group)), replace=False
            )
        )
        selected.append(group.iloc[chosen])
        coverage.append(
            {
                "outcome": outcome,
                "type": kind,
                "population": len(group),
                "sampled": len(chosen),
            }
        )
    return pd.concat(selected).reset_index(drop=True), pd.DataFrame(coverage)


def findings(frame, subgroups, minimum):
    """Descriptive findings only; no automated causal or superiority claims."""
    notes = [
        "Validation development evidence; subgroup comparisons are exploratory, not significance tests."
    ]
    extra = frame[frame.outcome == "smote_only_false_alarm"]
    gained = frame[frame.outcome == "smote_only_detected_fraud"]
    notes.append(
        f"At the saved cutoff, RF-SMOTE alone raises {len(extra):,} false alarms "
        f"and detects {len(gained):,} fraud cases (outcomes.csv)."
    )
    if len(extra):
        band = extra.amount_band.value_counts().idxmax()
        notes.append(
            f"The largest additional-false-alarm amount band is {band}: "
            f"{int((extra.amount_band == band).sum()):,} cases "
            "(outcomes_by_amount.csv). This count does not measure relative risk."
        )
    types = subgroups[
        (subgroups.grouping == "type")
        & (subgroups.model == "rf_smote")
        & (~subgroups.false_alarms_per_10000_sparse)
        & subgroups.false_alarms_per_10000.notna()
    ]
    if len(types):
        worst = types.loc[types.false_alarms_per_10000.idxmax()]
        notes.append(
            f"Among transaction types with at least {minimum} legitimate records, "
            f"RF-SMOTE's highest false-alarm rate is {worst['group']}: "
            f"{worst.false_alarms_per_10000:.2f} per 10,000 legitimate transactions "
            f"(denominator {int(worst.legitimate):,}; subgroups.csv)."
        )
    notes.append(
        "These patterns motivate checks of oversampling intensity and decision thresholds; "
        "they do not establish that synthetic samples caused individual errors."
    )
    return notes
