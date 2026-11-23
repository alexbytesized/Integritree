from fractions import Fraction
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from integritree.config import load_experiment
from integritree.ml.data import file_sha256
from integritree.ml.research import new_run
from investigation.diagnostics.analysis import metric_row, outcome_counts, sample_cases
from investigation.diagnostics.inputs import align_predictions, verify_links
from ratio_experiment.core import (
    choose_cutoff,
    make_pair,
    read_config,
    verify_training_ids,
)

ROOT = Path(__file__).resolve().parents[1]


def frame():
    codes = np.arange(8)
    return pd.DataFrame(
        {
            "source_row_number": codes + 1,
            "transaction_id": [f"hash:{i + 1}" for i in codes],
            "actual_label": codes // 4,
            "rf_risk_score": np.where(codes // 2 % 2, 0.8, 0.2),
            "rf_smote_risk_score": np.where(codes % 2, 0.9, 0.1),
            "type": ["TRANSFER", "CASH_OUT"] * 4,
            "step": [1, 24, 25, 168, 169, 170, 171, 172],
            "amount": [0, 999, 1000, 9999, 10000, 99999, 100000, 1000000],
            "is_zero_amount": [1, 0, 0, 0, 0, 0, 0, 0],
            "is_merchant_origin": 0,
            "is_merchant_dest": 0,
            "split": "validation",
            "run_id": "reference",
        }
    )


def test_config_paths_and_fixed_protocol(tmp_path):
    path = tmp_path / "config.yaml"
    settings = yaml.safe_load((ROOT / "config.yaml").read_text())
    path.write_text(yaml.safe_dump(settings))
    loaded = read_config(path)
    assert loaded["models"] == (tmp_path / settings["models"]).resolve()
    assert loaded["ratio_denominators"] == [200, 300, 500]
    for key, value in (
        ("ratio_denominators", [100, 200]),
        ("seed", 43),
        ("sample_per_group", 6),
    ):
        broken = settings | {key: value}
        path.write_text(yaml.safe_dump(broken))
        with pytest.raises(ValueError):
            read_config(path)


@pytest.mark.parametrize(
    "train,valid", [([1, 1], [2]), ([1, None], [2]), ([1], [1]), ([1], [2, 2])]
)
def test_identity_rejection(train, valid):
    with pytest.raises(ValueError):
        verify_training_ids(train, valid)


def test_all_outcomes_bands_and_preserved_reference():
    original = frame()
    before = original.copy(deep=True)
    pair = make_pair(original, original.rf_smote_risk_score, 0.43)
    assert outcome_counts(pair).rows.tolist() == [1] * 8
    assert pair.amount_band.cat.codes.tolist() == [0, 0, 1, 1, 2, 2, 3, 4]
    pd.testing.assert_frame_equal(before, original)
    assert pair.threshold.nunique() == 1
    for name in ("rf", "rf_smote", "reference_100"):
        np.testing.assert_array_equal(
            pair[f"{name}_predicted_label"], pair[f"{name}_risk_score"] >= 0.43
        )


def test_exact_shared_threshold_oracle_and_ties():
    rng = np.random.default_rng(42)
    labels = np.r_[np.zeros(100, dtype=int), np.ones(15, dtype=int)]
    rf, smote = rng.integers(0, 101, (2, len(labels))) / 100
    reference = pd.DataFrame({"actual_label": labels, "rf_risk_score": rf})
    chosen, grid = choose_cutoff(reference, smote)
    expected = []
    for percent in range(1, 101):
        fractions = []
        for name, scores in (("rf", rf), ("rf_smote", smote)):
            p = scores >= percent / 100
            tp, fp, fn = (
                int((p & (labels == 1)).sum()),
                int((p & (labels == 0)).sum()),
                int((~p & (labels == 1)).sum()),
            )
            row = grid.iloc[percent - 1]
            assert [row[f"{name}_{x}"] for x in ("tp", "fp", "fn")] == [tp, fp, fn]
            fractions.append(Fraction(2 * tp, 2 * tp + fp + fn))
        expected.append(((sum(fractions) / 2, -abs(percent - 50), percent), percent))
    assert chosen["threshold"] == max(expected)[1] / 100
    assert chosen["mean_f1_fraction"] == str(max(expected)[0][0])
    flat = pd.DataFrame({"actual_label": [0, 1], "rf_risk_score": [0.0, 0.0]})
    assert choose_cutoff(flat, [0.0, 0.0])[0]["threshold"] == 0.5
    # Best intervals 49% and 51%, equally close: prefer 51%.
    from integritree.ml.threshold_search import percent_grid_thresholds

    y = [1, 1, 0, 0]
    # Independently brute-check every tie, including score equality at a grid point.
    tie_result, tie_grid = percent_grid_thresholds(
        y, [0.49, 0.51, 0.5, 0.48], [0.51, 0.49, 0.48, 0.5]
    )
    winner = max(
        tie_grid.itertuples(),
        key=lambda r: (
            Fraction(2 * r.rf_tp, 2 * r.rf_tp + r.rf_fp + r.rf_fn)
            + Fraction(
                2 * r.rf_smote_tp, 2 * r.rf_smote_tp + r.rf_smote_fp + r.rf_smote_fn
            ),
            -abs(r.threshold_percent - 50),
            r.threshold_percent,
        ),
    )
    assert tie_result["threshold"] == winner.threshold


def test_sparse_undefined_metrics_and_deterministic_samples():
    pair = make_pair(frame(), frame().rf_smote_risk_score, 0.43)
    policy = load_experiment(ROOT.parent / "configs" / "experiment.yaml").evaluation
    single = pair[pair.actual_label == 0]
    metrics = metric_row(single, "rf_smote", 0.43, policy, 20)
    assert metrics["recall"] is None and metrics["recall_sparse"]
    assert metrics["precision_denominator"] == 2
    first, _ = sample_cases(pair, 1, 42)
    second, _ = sample_cases(pair.sample(frac=1, random_state=10), 1, 42)
    assert first.source_row_number.tolist() == second.source_row_number.tolist()


def test_duplicate_prediction_and_test_provenance_rejection(tmp_path):
    pair = make_pair(frame(), frame().rf_smote_risk_score, 0.43)
    pair.loc[1, "transaction_id"] = pair.loc[0, "transaction_id"]
    with pytest.raises(ValueError, match="Duplicate"):
        align_predictions(
            pair, np.arange(1, 9), pair.actual_label, "hash", "reference", 0.43
        )
    with pytest.raises(ValueError, match="validation"):
        verify_links({"split": "test", "official_test": True}, None, tmp_path, tmp_path)


def test_existing_run_never_overwritten(tmp_path):
    run = new_run(tmp_path, "ratio", "unique")
    marker = run / "marker"
    marker.write_text("keep")
    before = file_sha256(marker)
    with pytest.raises(FileExistsError):
        new_run(tmp_path, "ratio", "unique")
    assert file_sha256(marker) == before


def test_failed_report_is_explicit_and_no_training(tmp_path):
    from ratio_experiment.runner import run

    config = yaml.safe_load((ROOT / "config.yaml").read_text())
    report = tmp_path / "test_report"
    report.mkdir()
    (report / "metadata.json").write_text(
        json.dumps({"split": "test", "official_test": True})
    )
    config["report"] = str(report)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="validation"):
        run(tmp_path, config_path, "rejected")
    output = tmp_path / "outputs" / "rejected"
    meta = json.loads((output / "manifest.json").read_text())
    assert meta["status"] == "failed" and meta["new_fits_started"] == 0
    assert "incomplete" in (output / "report.html").read_text()
