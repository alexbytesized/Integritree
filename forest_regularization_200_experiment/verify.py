"""Verify saved evaluation evidence without refitting or rerunning explanations."""

import argparse
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path[:0] = [
    str(ROOT.parent / "backend/src"),
    str(ROOT.parent / "backend/forest_regularization_experiment"),
]

import numpy as np
import pandas as pd
from compare_cutoffs import verify_grid
from evaluate_csv import check_counts, load_csv
from experiment import read_json, time_summary, verify_unchanged
from integritree.config import EvaluationConfig
from integritree.ml.data import file_sha256
from integritree.ml.evaluation import metrics
from integritree.ml.features import FEATURE_COLUMNS


def verify_results(output, manifest):
    _, labels, ids = load_csv(Path(manifest["settings"]["validation_csv"]))
    policy = EvaluationConfig.model_validate(
        manifest["baseline_configuration"]["evaluation"]
    )
    summary = pd.read_csv(output / "comparison.csv", float_precision="round_trip")
    shap_summary = pd.read_csv(output / "time_shap.csv", float_precision="round_trip")
    cohorts = pd.read_csv(output / "shap_cohorts.csv")
    if len(summary) != 24 or len(shap_summary) != 48:
        raise ValueError("Incomplete summary tables")
    if cohorts.duplicated(["cohort", "source_row_number"]).any():
        raise ValueError("Duplicate cohort membership")
    np.testing.assert_array_equal(
        cohorts.actual_label, labels[cohorts.source_row_number.to_numpy() - 1]
    )
    chosen = np.unique(cohorts.source_row_number)
    sensitivity = pd.read_csv(
        output / "sensitivity_cohort.csv"
    ).source_row_number.to_numpy()
    checks = []
    for item in manifest["plan"]:
        name = item["candidate"]
        folder = output / "evaluation" / name
        frame = pd.read_parquet(folder / "predictions.parquet")
        np.testing.assert_array_equal(frame.actual_label, labels)
        np.testing.assert_array_equal(frame.source_row_number, ids)
        decision = read_json(folder / "decision.json")
        grid = pd.read_csv(folder / "thresholds.csv", float_precision="round_trip")
        verify_grid(
            labels,
            {kind: frame[f"{kind}_score"].to_numpy() for kind in ("rf", "rf_smote")},
            decision,
            grid,
        )
        for kind in ("rf", "rf_smote"):
            for mode, cutoff in (
                ("fixed_43", 0.43),
                ("selected", decision["threshold"]),
            ):
                matches = summary[
                    (summary.candidate == name)
                    & (summary.model == kind)
                    & (summary["mode"] == mode)
                ]
                if len(matches) != 1:
                    raise ValueError("Missing or duplicate metric row")
                row = matches.iloc[0].to_dict()
                if row["threshold"] != cutoff:
                    raise ValueError("Incorrect cutoff in report")
                scores = frame[f"{kind}_score"].to_numpy()
                check_counts(row, labels, scores)
                np.testing.assert_array_equal(
                    frame[f"{kind}_{mode}_label"], scores >= cutoff
                )
                recalculated = metrics(labels, scores, cutoff, policy)
                for metric in ("precision", "recall", "f1", "pr_auc"):
                    expected = recalculated["metrics"][metric]["value"]
                    if expected is None:
                        if not pd.isna(row[metric]):
                            raise ValueError("Undefined metric mismatch")
                    else:
                        np.testing.assert_allclose(
                            row[metric], expected, rtol=0, atol=1e-12
                        )
            explanation = pd.read_parquet(folder / f"{kind}_shap.parquet")
            np.testing.assert_array_equal(explanation.source_row_number, chosen)
            expected_scores = frame[f"{kind}_score"].to_numpy()[chosen - 1]
            np.testing.assert_array_equal(explanation.score, expected_scores)
            error = float(
                np.max(
                    np.abs(
                        explanation.base_score
                        + explanation[FEATURE_COLUMNS].sum(axis=1)
                        - expected_scores
                    )
                )
            )
            if error > 1e-6:
                raise ValueError("Saved SHAP reconstruction failed")
            for cohort, members in cohorts.groupby("cohort"):
                phi = (
                    explanation.set_index("source_row_number")
                    .loc[members.source_row_number, FEATURE_COLUMNS]
                    .to_numpy()
                )
                actual = shap_summary[
                    (shap_summary.candidate == name)
                    & (shap_summary.model == kind)
                    & (shap_summary.cohort == cohort)
                ].iloc[0]
                for field, expected in time_summary(phi).items():
                    if expected is None:
                        if not pd.isna(actual[field]):
                            raise ValueError("Undefined SHAP summary mismatch")
                    else:
                        np.testing.assert_allclose(
                            actual[field], expected, atol=1e-12, rtol=0
                        )
            sweep = pd.read_parquet(folder / f"{kind}_time_sweep.parquet")
            np.testing.assert_array_equal(sweep.source_row_number.unique(), sensitivity)
            if (
                sweep.groupby("source_row_number").size().ne(168).any()
                or sweep.duplicated(
                    ["source_row_number", "simulation_hour", "simulation_day"]
                ).any()
            ):
                raise ValueError("Incomplete hour/day sweep")
            np.testing.assert_array_equal(
                np.sort(sweep.simulation_hour.unique()), np.arange(24)
            )
            np.testing.assert_array_equal(
                np.sort(sweep.simulation_day.unique()), np.arange(7)
            )
            limits = sweep.groupby("source_row_number").score.agg(["min", "max"])
            per_row = pd.read_csv(
                folder / f"{kind}_time_sensitivity.csv", float_precision="round_trip"
            ).set_index("source_row_number")
            np.testing.assert_allclose(
                per_row.loc[limits.index, "range_pp"],
                100 * (limits["max"] - limits["min"]),
                rtol=0,
                atol=1e-12,
            )
            for cutoff, column in (
                (0.43, "crosses_43_percent"),
                (decision["threshold"], "crosses_selected_cutoff"),
            ):
                np.testing.assert_array_equal(
                    per_row.loc[limits.index, column],
                    (limits["min"] < cutoff) & (limits["max"] >= cutoff),
                )
            checks.append(
                {
                    "candidate": name,
                    "model": kind,
                    "rows": len(labels),
                    "shap_rows": len(chosen),
                    "shap_max_error": error,
                    "time_sweep_scores": len(sweep),
                }
            )
    for (_, _), pair in summary.groupby(["candidate", "mode"]):
        np.testing.assert_allclose(
            pair.pair_mean_f1, pair.f1.mean(), atol=1e-12, rtol=0
        )
    for _, pair in summary.groupby("candidate"):
        if (
            pair.query("mode == 'selected'").f1.mean() + 1e-12
            < pair.query("mode == 'fixed_43'").f1.mean()
        ):
            raise ValueError("Selected mean F1 is worse than an included cutoff")
    return {
        "status": "passed",
        "models": checks,
        "rows": len(labels),
        "actual_fraud": int(labels.sum()),
        "thresholds_verified": 600,
        "new_fits_completed": manifest["new_fits_completed"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    output = args.run.resolve()
    manifest = read_json(output / "manifest.json")
    if manifest["status"] != "complete":
        raise ValueError("Only completed runs can be independently verified")
    for name, digest in manifest["files"].items():
        path = (output / name).resolve()
        if not path.is_relative_to(output) or file_sha256(path) != digest:
            raise ValueError("Output fingerprint mismatch")
    verify_unchanged(manifest["input_sha256"])
    verify_unchanged(manifest["implementation_sha256"])
    result = verify_results(output, manifest)
    print(
        f"Verified {len(result['models'])} models, {result['thresholds_verified']} cutoffs, and source/output fingerprints"
    )


if __name__ == "__main__":
    main()
