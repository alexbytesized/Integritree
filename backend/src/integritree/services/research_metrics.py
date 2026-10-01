"""Disk-backed adapter for the established evaluation policy.

SQLite sorts tied scores on disk; neither exact AP nor exports materialize the
uploaded population. Parity with ml.evaluation.evaluate_pair is tested.
"""

import math
from scipy.stats import binomtest, chi2
from integritree.ml.evaluation import METRICS, comparison, measured, validate_policy


def evaluate_database(db, threshold, policy):
    validate_policy(policy)
    rows, positives = db.execute("SELECT count(*), sum(label) FROM records").fetchone()
    models = {}
    for name in ("rf", "rf_smote"):
        tn, fp, fn, tp = (
            db.execute(
                f"SELECT count(*) FROM records WHERE label=? AND {name}_pred=?", (y, p)
            ).fetchone()[0]
            for y, p in ((0, 0), (0, 1), (1, 0), (1, 1))
        )
        seen = found = 0
        ap = 0.0
        for count, fraud in db.execute(
            f"SELECT count(*), sum(label) FROM records GROUP BY {name}_score ORDER BY {name}_score DESC"
        ):
            seen += count
            found += fraud
            ap += (fraud / positives) * (found / seen) if positives else 0

        def ratio(n, d, reason):
            return measured(n / d) if d else measured(None, reason)

        values = {
            "precision": ratio(tp, tp + fp, "no_predicted_positives"),
            "recall": ratio(tp, tp + fn, "no_actual_positives"),
            "f1": ratio(2 * tp, 2 * tp + fp + fn, "no_actual_or_predicted_positives"),
            "mcc": ratio(
                tp * tn - fp * fn,
                math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)),
                "zero_mcc_denominator",
            ),
            "pr_auc": measured(ap)
            if positives
            else measured(None, "no_actual_positives"),
            "accuracy": measured((tp + tn) / rows),
        }
        models[name] = {
            "rows": rows,
            "actual_fraud": positives,
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
    a, b, c, d = (
        db.execute(
            "SELECT count(*) FROM records WHERE (rf_pred=label)=? AND (rf_smote_pred=label)=?",
            pair,
        ).fetchone()[0]
        for pair in ((1, 1), (1, 0), (0, 1), (0, 0))
    )
    discordant = b + c
    statistic = (abs(b - c) - 1) ** 2 / discordant if discordant else None
    p = float(chi2.sf(statistic, 1)) if discordant else 1.0
    test = {
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
        "status": "computed" if discordant else "No discordant pairs",
        "p_value_convention": None if discordant else "1_when_no_discordant_pairs",
        "approximation_caution": 1 <= discordant <= 24,
        "decision": "reject_null" if p < policy.alpha else "fail_to_reject_null",
        "lower_observed_error_model": "rf_smote" if c > b else "rf" if b > c else None,
        "scope": "paired classification error rates; not significance of individual metrics",
        "supplementary": {
            "method": "exact_binomial_two_sided",
            "p_value": float(binomtest(b, discordant, 0.5).pvalue),
            "reason": "1_to_24_discordant_pairs",
            "role": "supplementary",
        }
        if 1 <= discordant <= 24
        else None,
    }
    return {
        "models": models,
        "statistical_test": test,
        "descriptive_comparisons": {
            k: comparison(
                models["rf"]["metrics"][k]["value"],
                models["rf_smote"]["metrics"][k]["value"],
                k,
                policy.percentage_difference,
            )
            for k in METRICS
        },
    }
