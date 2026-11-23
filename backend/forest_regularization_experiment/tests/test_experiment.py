"""Synthetic checks for isolation, paired comparisons, and honest diagnostics."""

from pathlib import Path
from types import SimpleNamespace

import experiment as exp
import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.ensemble import RandomForestClassifier

from integritree.config import load_experiment
from integritree.ml.artifacts import ModelBundle, load_bundle
from integritree.ml.features import FEATURE_COLUMNS
from integritree.ml.preprocessing import FittedPreprocessor

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "change",
    [
        {"forests": [[10, 1], [12, 1]]},
        {"forests": [[10, 1], [8, 0]]},
        {"forests": [[10, 1], [8, True]]},
        {"forests": [[10, 1], [10, 1]]},
        {"seed": 123},
        {"background_rows": 100},
        {"reuse_existing": "yes"},
        {"test_split": "test"},
    ],
)
def test_rejects_confounded_or_invalid_protocol(tmp_path, change):
    settings = yaml.safe_load((ROOT / "config.yaml").read_text()) | change
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(settings))
    with pytest.raises(ValueError):
        exp.read_config(path)


def test_time_summary_distinguishes_absolute_net_and_zero():
    phi = np.zeros((2, len(FEATURE_COLUMNS)))
    phi[:, 0], phi[:, 1], phi[:, 7] = 0.2, -0.15, 0.05
    summary = exp.time_summary(phi)
    assert summary["mean_time_abs_pp"] == pytest.approx(35)
    assert summary["mean_time_net_abs_pp"] == pytest.approx(5)
    assert summary["time_share_abs"] == pytest.approx(0.875)
    assert summary["time_strictly_largest_group_fraction"] == 1
    assert exp.time_summary(np.zeros_like(phi))["time_share_abs"] is None
    assert exp.time_summary(phi[:0])["rows"] == 0


def test_cohorts_are_reproducible_and_based_only_on_baseline():
    settings = exp.read_config(ROOT / "config.yaml")
    y = np.array([0, 1] * 15)
    scores = {"rf": np.linspace(0, 1, 30), "rf_smote": np.zeros(30)}
    a, b = exp.make_cohorts(y, scores, settings), exp.make_cohorts(y, scores, settings)
    for name in a:
        np.testing.assert_array_equal(a[name], b[name])
    assert not len(a["baseline_flagged_rf_smote"])
    assert np.all(y[a["actual_fraud"]] == 1)
    assert np.all(scores["rf"][a["baseline_flagged_rf"]] >= 0.43)


def test_time_sweep_preserves_non_time_inputs_and_uses_saved_scaler():
    class TimeModel:
        classes_ = np.array([0, 1])

        def predict_proba(self, values):
            # The two non-time transaction values must remain fixed.
            assert set(values[:, 7]) == {0.2, 0.8}
            p = np.round((values[:, 0] - 0.1) / 0.1) / 23
            return np.column_stack([1 - p, p])

    preprocessor = SimpleNamespace(
        state=SimpleNamespace(
            scaled_columns=["log_amount", "hour_of_day", "day_of_week"],
            scale=[0.2, 0.1, 0.2],
            offset=[0, 0.1, 0.3],
        )
    )
    values = np.zeros((2, len(FEATURE_COLUMNS)))
    values[:, 7] = [0.2, 0.8]
    grid, per_row = exp.time_sweep(
        TimeModel(), values, np.array([10, 20]), preprocessor
    )
    assert len(grid) == 336
    assert grid.groupby("source_row_number").size().tolist() == [168, 168]
    np.testing.assert_allclose(per_row.range_pp, [100, 100])
    assert per_row.crosses_43_percent.all()


def synthetic_inputs():
    rng = np.random.default_rng(42)
    inputs = pd.DataFrame(
        {
            "step": rng.integers(1, 744, 1360),
            "type": rng.choice(["TRANSFER", "CASH_OUT"], 1360),
            "amount": rng.uniform(1, 10000, 1360),
            "nameOrig": "C_TEST",
            "nameDest": "C_DEST",
        }
    )
    prep = FittedPreprocessor.fit(inputs.iloc[:1200])
    features = prep.transform(inputs)
    y = np.zeros(1360, dtype="uint8")
    y[np.argsort(inputs.amount.iloc[:1200])[-8:]] = 1
    y[1200::10] = 1
    config = load_experiment(ROOT.parent / "configs" / "experiment.yaml").model_copy(
        deep=True
    )
    config.random_forest.n_estimators = 4
    config.random_forest.max_depth = 10
    config.smote.sampling_ratio = 0.01
    config.scoring.threshold = 0.43
    parameters = exp.forest_parameters(config, 1)
    models = {
        name: RandomForestClassifier(**parameters).fit(
            features.iloc[:1200].to_numpy(), y[:1200]
        )
        for name in exp.KINDS
    }
    bundle = ModelBundle(
        models, prep, config, {"prepared_metadata_sha256": "synthetic-only"}
    )
    return features, y, bundle


def test_complete_synthetic_execution_and_no_test_access(tmp_path, monkeypatch):
    features, truth, bundle = synthetic_inputs()
    settings = yaml.safe_load((ROOT / "config.yaml").read_text())
    settings.update(
        forests=[[10, 1], [6, 10]],
        shap_uniform_rows=12,
        shap_fraud_rows=4,
        shap_baseline_flagged_rows=4,
        sensitivity_rows=3,
    )
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(settings))
    source = tmp_path / "untouched.txt"
    source.write_text("preserved")
    before = exp.file_sha256(source)
    plan = [
        {
            "candidate": "depth_10_leaf_1",
            "max_depth": 10,
            "min_samples_leaf": 1,
            "source": "synthetic",
            "new_fits": 0,
        },
        {
            "candidate": "depth_6_leaf_10",
            "max_depth": 6,
            "min_samples_leaf": 10,
            "source": None,
            "new_fits": 2,
        },
    ]
    monkeypatch.setattr(
        exp,
        "prepare_plan",
        lambda _: (bundle, plan, {str(source): before}, {"passed": True}),
    )
    accessed = []

    def loader(path, split):
        accessed.append(split)
        assert split in ("train", "validation")
        positions = np.arange(1200) if split == "train" else np.arange(1200, 1360)
        return (
            features.iloc[positions],
            pd.Series(truth[positions]),
            pd.Series(positions + 1),
        )

    monkeypatch.setattr(exp, "load_prepared_split", loader)
    output = exp.run(tmp_path, config, "synthetic_execution", execute=True)
    manifest = exp.read_json(output / "manifest.json")
    assert manifest["status"] == "complete"
    assert manifest["fits_started"] == 2
    assert manifest["test_used"] is False
    assert accessed == ["train", "validation"]
    assert exp.file_sha256(source) == before
    table = pd.read_csv(output / "metrics.csv", float_precision="round_trip")
    assert len(table) == 4 and table.threshold.eq(0.43).all()
    assert table.rows.eq(160).all()
    for name in ("depth_10_leaf_1", "depth_6_leaf_10"):
        p = pd.read_parquet(output / name / "predictions.parquet")
        np.testing.assert_array_equal(p.source_row_number, np.arange(1201, 1361))
        for model in exp.KINDS:
            phi = pd.read_parquet(output / name / f"{model}_shap.parquet")
            np.testing.assert_allclose(
                phi.base_score + phi[FEATURE_COLUMNS].sum(axis=1), phi.score, atol=1e-6
            )
            np.testing.assert_array_equal(
                p[f"{model}_predicted_label"], p[f"{model}_score"] >= 0.43
            )
    for name, fingerprint in manifest["files"].items():
        assert exp.file_sha256(output / name) == fingerprint
    with pytest.raises(ValueError):
        load_bundle(output / "depth_6_leaf_10")
    with pytest.raises(FileExistsError):
        exp.run(tmp_path, config, "synthetic_execution", execute=True)


def test_preflight_never_loads_training_arrays_or_fits(tmp_path, monkeypatch):
    _, _, bundle = synthetic_inputs()
    plan = [
        {"candidate": "depth_10_leaf_1", "new_fits": 0},
        {"candidate": "depth_8_leaf_1", "new_fits": 2},
    ]
    monkeypatch.setattr(
        exp, "prepare_plan", lambda _: (bundle, plan, {}, {"passed": True})
    )

    def forbidden(*args, **kwargs):
        pytest.fail("Preflight must not train or read split arrays")

    monkeypatch.setattr(exp, "load_prepared_split", forbidden)
    monkeypatch.setattr(RandomForestClassifier, "fit", forbidden)
    output = exp.run(tmp_path, ROOT / "config.yaml", "check", execute=False)
    manifest = exp.read_json(output / "manifest.json")
    assert manifest["status"] == "preflight_complete"
    assert manifest["fits_started"] == 0


def test_reuse_rejects_changed_ratio_or_parameters():
    _, _, reference = synthetic_inputs()
    changed = reference.config.model_copy(deep=True)
    changed.smote.sampling_ratio = 0.02
    candidate = ModelBundle(
        reference.models, reference.preprocessor, changed, reference.metadata
    )
    with pytest.raises(ValueError, match="smote"):
        exp.verify_reuse(
            candidate, reference, exp.forest_parameters(reference.config, 1)
        )
    with pytest.raises(ValueError, match="parameters"):
        exp.verify_reuse(
            reference,
            reference,
            exp.forest_parameters(reference.config, 1) | {"max_depth": 6},
        )
