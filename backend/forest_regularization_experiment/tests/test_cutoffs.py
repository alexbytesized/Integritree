"""Cutoff-only analysis: score ties, exact optima, preservation, and no fitting."""

from pathlib import Path

import compare_cutoffs as cutoff
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from integritree.config import load_experiment


def test_grid_ties_prefer_50_and_equal_scores_predict_fraud():
    y = np.array([0, 1])
    scores = {"rf": np.array([0.0, 1.0]), "rf_smote": np.array([0.0, 1.0])}
    decision, grid = cutoff.percent_grid_thresholds(
        y, rf_scores=scores["rf"], smote_scores=scores["rf_smote"]
    )
    assert decision["threshold"] == 0.5
    assert cutoff.verify_grid(y, scores, decision, grid)["status"] == "passed"
    assert grid.iloc[-1].rf_tp == 1  # score == 100% must be included
    scores["rf"][1] = 0.43
    decision, grid = cutoff.percent_grid_thresholds(y, scores["rf"], scores["rf_smote"])
    assert grid.loc[grid.threshold_percent == 43, "rf_tp"].item() == 1
    assert grid.loc[grid.threshold_percent == 44, "rf_tp"].item() == 0
    cutoff.verify_grid(y, scores, decision, grid)


def test_verifier_rejects_changed_grid_and_nonoptimal_decision():
    y = np.array([0, 1, 0, 1])
    scores = {
        "rf": np.array([0.1, 0.8, 0.2, 0.9]),
        "rf_smote": np.array([0.2, 0.7, 0.1, 0.8]),
    }
    decision, grid = cutoff.percent_grid_thresholds(y, scores["rf"], scores["rf_smote"])
    with pytest.raises(ValueError, match="selection"):
        cutoff.verify_grid(y, scores, decision | {"threshold": 0.99}, grid)
    grid.loc[0, "rf_tp"] = 0
    with pytest.raises(ValueError, match="count mismatch"):
        cutoff.verify_grid(y, scores, decision, grid)


def build_source(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    config = load_experiment(
        Path(__file__).resolve().parents[2] / "configs/experiment.yaml"
    )
    plan = [
        {"candidate": "depth_10_leaf_1", "max_depth": 10, "min_samples_leaf": 1},
        {"candidate": "depth_8_leaf_1", "max_depth": 8, "min_samples_leaf": 1},
    ]
    cutoff.write_json(
        source / "plan.json",
        {
            "candidates": plan,
            "fixed_configuration": {
                "scoring": {"threshold": 0.43},
                "evaluation": config.evaluation.model_dump(),
            },
        },
    )
    y = np.array([0, 1, 1, 0, 1, 0])
    rows = []
    for i, item in enumerate(plan):
        folder = source / item["candidate"]
        folder.mkdir()
        frame = pd.DataFrame({"source_row_number": np.arange(1, 7), "actual_label": y})
        for kind, scores in (
            ("rf", np.array([0.1, 0.43, 0.7, 0.4, 0.9, 0.2])),
            ("rf_smote", np.array([0.2, 0.5, 0.65, 0.6, 0.85, 0.15])),
        ):
            scores = scores * (1 if not i else 0.7)
            frame[f"{kind}_score"] = scores
            frame[f"{kind}_predicted_label"] = (scores >= 0.43).astype(int)
            rows.append(
                cutoff.metric_row(
                    y,
                    scores,
                    0.43,
                    config.evaluation,
                    item["candidate"],
                    kind,
                    "fixed_43",
                )
            )
            pd.DataFrame(
                {"source_row_number": [1, 1, 2, 2], "score": [0.1, 0.8, 0.5, 0.9]}
            ).to_parquet(folder / f"{kind}_time_sweep.parquet")
        frame.to_parquet(folder / "predictions.parquet", index=False)
    pd.DataFrame(rows).to_csv(source / "metrics.csv", index=False, float_format="%.17g")
    manifest = {
        "kind": "isolated_forest_regularization",
        "status": "complete",
        "source_preservation_verified": True,
        "test_used": False,
        "plan": plan,
        "files": {
            p.relative_to(source).as_posix(): cutoff.file_sha256(p)
            for p in source.rglob("*")
            if p.is_file()
        },
    }
    cutoff.write_json(source / "manifest.json", manifest)
    return source


def test_complete_followup_reuses_scores_and_never_trains(tmp_path, monkeypatch):
    source = build_source(tmp_path)
    before = {str(p): cutoff.file_sha256(p) for p in source.rglob("*") if p.is_file()}

    def forbidden(*args, **kwargs):
        pytest.fail("Cutoff analysis must not train models")

    monkeypatch.setattr(RandomForestClassifier, "fit", forbidden)
    output = cutoff.run(source, tmp_path / "outputs", "synthetic_cutoffs")
    metadata = cutoff.read_json(output / "manifest.json")
    assert metadata["status"] == "complete"
    assert metadata["new_fits"] == 0 and metadata["test_used"] is False
    cutoff.verify_hashes(before)
    for name, digest in metadata["files"].items():
        assert cutoff.file_sha256(output / name) == digest
    summary = pd.read_parquet(output / "pair_summary.parquet")
    assert (summary.mean_f1_selected >= summary.mean_f1_fixed).all()
    assert len(pd.read_parquet(output / "threshold_grid.parquet")) == 200
    chosen = pd.read_parquet(output / "metrics_selected.parquet")
    fixed = pd.read_parquet(output / "metrics_fixed_43.parquet")
    np.testing.assert_array_equal(chosen.pr_auc, fixed.pr_auc)
    for row in summary.itertuples(index=False):
        old = pd.read_parquet(source / row.candidate / "predictions.parquet")
        new = pd.read_parquet(output / row.candidate / "predictions_selected.parquet")
        np.testing.assert_array_equal(new.source_row_number, old.source_row_number)
        for kind in ("rf", "rf_smote"):
            np.testing.assert_array_equal(new[f"{kind}_score"], old[f"{kind}_score"])
            np.testing.assert_array_equal(
                new[f"{kind}_selected_label"], old[f"{kind}_score"] >= row.threshold
            )
    with pytest.raises(FileExistsError):
        cutoff.run(source, tmp_path / "outputs", "synthetic_cutoffs")


@pytest.mark.parametrize("invalid", ["incomplete", "test", "corrupt"])
def test_rejects_unverified_source(tmp_path, invalid):
    source = build_source(tmp_path)
    manifest = cutoff.read_json(source / "manifest.json")
    if invalid == "incomplete":
        manifest["status"] = "running"
    elif invalid == "test":
        manifest["test_used"] = True
    else:
        (source / "metrics.csv").write_text("corrupt")
    cutoff.write_json(source / "manifest.json", manifest)
    with pytest.raises(ValueError):
        cutoff.source_inventory(source)
