"""Independent acceptance checks for a completed investigation (not a pytest test)."""

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["MPLCONFIGDIR"] = str(ROOT / "runtime" / "matplotlib")
os.environ["MPLBACKEND"] = "Agg"

from diagnostics.inputs import verify_unchanged
from diagnostics.reporting import verify_offline_report
from integritree.ml.data import file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--expect-current-counts", action="store_true")
    args = parser.parse_args()
    output = args.run.resolve()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["split"] == "validation" and manifest["test_used"] is False
    assert (
        manifest["models_retrained"] is False
        and manifest["thresholds_changed"] is False
    )
    verify_unchanged(manifest["input_sha256"])
    for name, expected in manifest["files"].items():
        path = (output / name).resolve()
        assert path.is_relative_to(output)
        assert file_sha256(path) == expected, name
    verify_offline_report(output)
    frame = pd.read_parquet(output / "validation_details.parquet")
    outcomes = (
        pd.read_csv(output / "tables" / "outcomes.csv")
        .set_index("outcome")
        .rows.to_dict()
    )
    assert sum(outcomes.values()) == manifest["rows"] == len(frame)
    if args.expect_current_counts:
        assert manifest["model_run_id"] == "paysim_three_stage_20260928_172539"
        assert manifest["rows"] == 636262
        assert outcomes == {
            "both_correct_legitimate": 634994,
            "smote_only_false_alarm": 397,
            "rf_only_false_alarm": 0,
            "both_false_alarm": 49,
            "both_missed_fraud": 489,
            "smote_only_detected_fraud": 81,
            "rf_only_detected_fraud": 0,
            "both_detected_fraud": 252,
        }
    sample = pd.read_csv(
        output / "tables" / "sampled_cases.csv", float_precision="round_trip"
    )
    contributions = pd.read_csv(
        output / "tables" / "shap_contributions.csv", float_precision="round_trip"
    )
    assert set(contributions.source_row_number) == set(sample.source_row_number)
    tolerance = manifest["shap_policy"]["additivity_tolerance"]
    for (row_id, model), group in contributions.groupby(["source_row_number", "model"]):
        assert len(group) == 11
        row = sample.set_index("source_row_number").loc[row_id]
        assert (
            abs(
                group.base_value.iloc[0]
                + group.contribution.sum()
                - group.output_value.iloc[0]
            )
            <= tolerance
        )
        assert abs(group.output_value.iloc[0] - row[f"{model}_risk_score"]) <= 1e-12
    assert contributions.groupby("source_row_number").model.nunique().eq(2).all()
    budgets = pd.read_csv(
        output / "tables" / "false_alarm_budgets.csv", float_precision="round_trip"
    )
    assert (budgets.fp <= budgets.false_alarm_budget).all()
    for row in budgets.itertuples():
        pred = (
            np.zeros(len(frame), dtype=bool)
            if row.no_alerts
            else frame[f"{row.model}_risk_score"] >= row.threshold
        )
        assert int((pred & frame.actual_label.eq(1)).sum()) == row.tp
        assert int((pred & frame.actual_label.eq(0)).sum()) == row.fp
    print(
        json.dumps(
            {
                "status": "verified",
                "rows": len(frame),
                "outcomes": outcomes,
                "sampled_transactions": len(sample),
                "explanations": manifest["paired_explanations"],
                "output_files_checked": len(manifest["files"]),
                "inputs_unchanged": True,
            }
        )
    )


if __name__ == "__main__":
    main()
