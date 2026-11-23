"""Independently check the completed full-data experiment without any new fit."""

from fractions import Fraction
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path[:0] = [str(ROOT), str(ROOT.parent)]
os.environ["MPLCONFIGDIR"] = str(ROOT / "runtime" / "matplotlib")
os.environ["MPLBACKEND"] = "Agg"

import numpy as np
import pandas as pd

from integritree.ml.data import file_sha256
from integritree.ml.preparation import load_prepared_split
from investigation.diagnostics.analysis import sample_cases
from investigation.diagnostics.reporting import verify_offline_report
from ratio_experiment.core import load_model, predict


def verify(output):
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete" and manifest["new_fits_started"] == 3
    assert not manifest["test_used"] and not manifest["deployment_changed"]
    for source in ("input_sha256", "implementation_sha256"):
        for name, expected in manifest[source].items():
            assert file_sha256(Path(name)) == expected, name
    for name, expected in manifest["files"].items():
        assert file_sha256(output / name) == expected, name
    verify_offline_report(output)
    prepared = Path(manifest["settings"]["prepared"])
    features, labels, ids = load_prepared_split(prepared, "validation")
    expected_counts = {
        "rf": (252, 49, 635391, 570),
        "reference_100": (333, 446, 634994, 489),
    }
    case_count = 0
    cutoffs = {}
    for denominator in (200, 300, 500):
        folder = output / f"ratio_{denominator}"
        verify_offline_report(folder)
        model = load_model(folder / "model", file_sha256(prepared / "metadata.json"))
        scores = predict(model, features)
        fixed = pd.read_parquet(folder / "predictions_43.parquet")
        selected = pd.read_parquet(folder / "predictions_selected.parquet")
        for frame in (fixed, selected):
            np.testing.assert_array_equal(frame.source_row_number, ids)
            np.testing.assert_array_equal(frame.actual_label, labels)
            assert frame.transaction_id.is_unique and frame.threshold.nunique() == 1
            np.testing.assert_allclose(
                frame.rf_smote_risk_score, scores, rtol=0, atol=1e-12
            )
            for name in ("rf", "rf_smote", "reference_100"):
                np.testing.assert_array_equal(
                    frame[f"{name}_predicted_label"],
                    frame[f"{name}_risk_score"] >= frame.threshold,
                )
        assert fixed.threshold.eq(0.43).all()
        y = fixed.actual_label.to_numpy()
        for name, expected in expected_counts.items():
            p = fixed[f"{name}_predicted_label"].to_numpy()
            actual = tuple(
                int(((y == a) & (p == b)).sum())
                for a, b in ((1, 1), (0, 1), (0, 0), (1, 0))
            )
            assert actual == expected, (name, actual)
        # Independent brute-force rational optimum; do not reuse the selection routine.
        ranks = []
        grid = pd.read_csv(folder / "tables" / "thresholds.csv")
        for percent in range(1, 101):
            f1s = []
            for name in ("rf", "rf_smote"):
                p = selected[f"{name}_risk_score"].to_numpy() >= percent / 100
                tp, fp, fn = (
                    int((p & (y == 1)).sum()),
                    int((p & (y == 0)).sum()),
                    int((~p & (y == 1)).sum()),
                )
                for key, value in (("tp", tp), ("fp", fp), ("fn", fn)):
                    assert grid.iloc[percent - 1][f"{name}_{key}"] == value
                f1s.append(Fraction(2 * tp, 2 * tp + fp + fn))
            ranks.append((sum(f1s) / 2, -abs(percent - 50), percent))
        cutoff = max(ranks)[2] / 100
        assert selected.threshold.eq(cutoff).all()
        cutoffs[str(denominator)] = cutoff
        expected_sample, _ = sample_cases(
            selected, manifest["settings"]["sample_per_group"], 42
        )
        saved_sample = pd.read_csv(folder / "tables" / "sampled_cases.csv")
        assert (
            saved_sample.source_row_number.tolist()
            == expected_sample.source_row_number.tolist()
        )
        for row in expected_sample.itertuples():
            for name in ("rf", "rf_smote"):
                item = json.loads(
                    (
                        folder
                        / "explanations"
                        / str(row.source_row_number)
                        / name
                        / "explanation.json"
                    ).read_text()
                )
                assert item["threshold"] == cutoff
                assert (
                    item["transaction_id"] == row.transaction_id
                    and item["model"] == name
                )
                assert (
                    abs(item["output_value"] - getattr(row, f"{name}_risk_score"))
                    <= 1e-12
                )
                assert (
                    abs(
                        item["base_value"]
                        + sum(f["contribution"] for f in item["features"])
                        - item["output_value"]
                    )
                    <= 1e-6
                )
                case_count += 1
        audit = json.loads((folder / "model" / "training_audit.json").read_text())
        assert audit["original_rows_preserved"] and not audit["validation_resampled"]
        assert audit["class_counts"]["1"] == int(
            audit["original_class_counts"]["0"] / denominator
        )
    return {
        "status": "verified",
        "validation_rows": len(labels),
        "fits": 3,
        "shared_cutoffs": cutoffs,
        "explanations": case_count,
        "full_validation_score_replay": True,
        "inputs_unchanged": True,
    }


if __name__ == "__main__":
    print(json.dumps(verify(Path(sys.argv[1]).resolve())))
