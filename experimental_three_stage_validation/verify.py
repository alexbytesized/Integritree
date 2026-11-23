"""Independently verify a saved complete or partial experiment."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from run import (KINDS, choose_forest, choose_ratio, file_sha256, metric_row,
                 read_json, verify_grid, ExperimentConfig, load_csv)


def verify(output):
    manifest = read_json(output / "manifest.json")
    if manifest["status"] not in ("complete", "incomplete_missing_models"):
        raise ValueError("Run is not a verified complete or missing-model result")
    assert manifest["training_fits"] == 0
    for name, digest in manifest["files"].items():
        assert file_sha256(output / name) == digest, name
    for name, digest in manifest["source_hashes"].items():
        assert file_sha256(Path(name)) == digest, name
    source = Path(manifest["source_run"])
    policy = ExperimentConfig.model_validate(read_json(source / "search_plan.json")["configuration"]).evaluation
    _, common_labels = load_csv(Path(manifest["validation_csv"]))
    recomputed = {}
    for key in manifest["completed_candidates"]:
        folder = output / "candidates" / key
        row = read_json(folder / "metrics.json")
        frame = pd.read_parquet(folder / "predictions.parquet")
        assert np.array_equal(frame.source_row_number, np.arange(1, manifest["rows"] + 1))
        labels = frame.actual_label.to_numpy()
        assert len(labels) == manifest["rows"] and int(labels.sum()) == manifest["actual_fraud"]
        assert np.array_equal(labels, common_labels)
        for kind in KINDS:
            expected = metric_row(labels, frame[f"{kind}_risk_score"].to_numpy(), 0.5, policy)
            assert expected == row[kind], key
        assert row["mean_ap"] == (row["rf"]["pr_auc"] + row["rf_smote"]["pr_auc"]) / 2
        recomputed[key] = row
    decisions = {}
    for stage, chooser, count in (("01_smote_ratio", choose_ratio, 8), ("02_random_forest", choose_forest, 12)):
        path = output / stage / "decision.json"
        if path.exists():
            decision = read_json(path)
            assert len(decision["candidates"]) == count
            assert all(row == recomputed[row["key"]] for row in decision["candidates"])
            assert chooser(decision["candidates"]) == decision["winner"]
            decisions[stage] = decision
    if "02_random_forest" in decisions:
        ratio = decisions["01_smote_ratio"]["winner"]["ratio"]
        assert all(row["ratio"] == ratio for row in decisions["02_random_forest"]["candidates"])
    result = {"status": manifest["status"], "candidates_verified": len(recomputed), "source_preservation_verified": True}
    if manifest["status"] == "complete":
        selected = manifest["selection"]
        assert decisions["02_random_forest"]["winner"]["key"] == selected["key"]
        frame = pd.read_parquet(output / "final_predictions.parquet")
        candidate = pd.read_parquet(output / "candidates" / selected["key"] / "predictions.parquet")
        pd.testing.assert_frame_equal(frame[candidate.columns], candidate)
        scores = {kind: frame[f"{kind}_risk_score"].to_numpy() for kind in KINDS}
        result.update(verify_grid(frame.actual_label.to_numpy(), scores,
                                 pd.read_parquet(output / "03_threshold/thresholds.parquet"), selected))
        final = read_json(output / "final_metrics.json")
        for kind in KINDS:
            assert np.array_equal(frame[f"{kind}_predicted_label"], scores[kind] >= selected["threshold"])
            assert final[kind] == {"model": kind, "threshold": selected["threshold"],
                                   **metric_row(frame.actual_label.to_numpy(), scores[kind], selected["threshold"], policy)}
    else:
        assert manifest["missing_candidates"]
        assert "selection" not in manifest
        assert not (output / "03_threshold").exists()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    print(verify(parser.parse_args().output.resolve()))
