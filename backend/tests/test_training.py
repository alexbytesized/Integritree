"""Synthetic end-to-end training checks; no real PaySim data or held-out evaluation."""
import json
import numpy as np
import pandas as pd
import pytest
from integritree.config import load_experiment
from integritree.settings import BACKEND_ROOT
from integritree.ml.data import RAW_COLUMNS, file_sha256
from integritree.ml.features import FEATURE_COLUMNS, SOURCE_COLUMNS
from integritree.ml.preparation import prepare_dataset, load_prepared_split
from integritree.ml.artifacts import load_bundle
from integritree.ml.inference import predict_records, predict_features
from integritree.ml.training import train_models, resample_training, validate_labels, main


@pytest.fixture
def prepared(tmp_path, raw_record):
    frame = pd.DataFrame([raw_record | {
        "step": i + 1, "amount": float(i + 1), "nameOrig": f"C_{i}",
        "type": "TRANSFER" if i % 2 else "CASH_OUT",
        "isFraud": int(i % 5 == 0),
    } for i in range(200)]).loc[:, RAW_COLUMNS]
    source = tmp_path / "synthetic.csv"
    frame.to_csv(source, index=False)
    config = load_experiment(BACKEND_ROOT / "configs/experiment.yaml").model_copy(deep=True)
    config.dataset = config.dataset.model_copy(update={
        "filename": source.name, "sha256": file_sha256(source), "row_count": len(frame)})
    config.random_forest.n_estimators = 4
    config.random_forest.max_depth = 3
    path = prepare_dataset(source, config, tmp_path / "prepared", "synthetic", progress=lambda _: None)
    return path, config


@pytest.mark.parametrize("comparison_method", ["absolute_over_mean", "signed_over_mean"])
def test_end_to_end_training_reload_pairing_and_no_held_out_reads(prepared, tmp_path, monkeypatch,
                                                               comparison_method):
    path, config = prepared
    # Method-only changes must not invalidate existing preparation or saved inference.
    config.evaluation.percentage_difference = comparison_method
    before = {name: file_sha256(path / name) for name in ("validation_features.parquet", "test_features.parquet")}
    import integritree.ml.training as training
    real_loader = training.load_prepared_split
    requested = []
    def training_only(bundle, split):
        requested.append(split)
        assert split == "train"
        return real_loader(bundle, split)
    monkeypatch.setattr(training, "load_prepared_split", training_only)
    output = train_models(path, config, tmp_path / "models", "first", progress=lambda _: None)
    bundle = load_bundle(output)
    assert bundle.config.evaluation.percentage_difference == comparison_method
    assert requested == ["train"]
    assert bundle.metadata["stage"] == "baseline_not_final"
    assert bundle.models["rf"].get_params() == bundle.models["rf_smote"].get_params()
    assert bundle.models["rf"].class_weight is None
    assert bundle.metadata["test_used"] is False
    audit = json.loads((output / "training_audit.json").read_text())
    assert audit["original_rows"] == 160
    assert audit["synthetic_rows"] == 96
    assert audit["class_counts"] == {"0": 128, "1": 128}
    for name, digest in before.items():
        assert file_sha256(path / name) == digest
    X, y, ids = load_prepared_split(path, "validation")
    identities = [f"{config.dataset.sha256}:{i}" for i in ids]
    batch = predict_features(bundle.models, X, identities, .5, "first")
    raw = pd.read_parquet(path / "source.parquet").set_index("source_row_number")
    inputs = raw.loc[ids, SOURCE_COLUMNS]
    original_state = bundle.preprocessor.state.model_dump()
    pd.testing.assert_frame_equal(batch, predict_records(bundle, inputs, identities))
    for i in range(len(inputs)):
        single = predict_records(bundle, inputs.iloc[[i]], [identities[i]])
        pd.testing.assert_frame_equal(single, batch.iloc[[i]].reset_index(drop=True))
    assert bundle.preprocessor.state.model_dump() == original_state
    again = train_models(path, config, tmp_path / "models", "second", progress=lambda _: None)
    other = load_bundle(again)
    reproduced = predict_features(other.models, X, identities, .5, "first")
    pd.testing.assert_frame_equal(batch, reproduced)
    manifest = (output / "metadata.json").read_bytes()
    with pytest.raises(FileExistsError):
        train_models(path, config, tmp_path / "models", "first", progress=lambda _: None)
    assert (output / "metadata.json").read_bytes() == manifest
    with (output / "rf.joblib").open("ab") as handle:
        handle.write(b"corruption")
    with pytest.raises(ValueError, match="fingerprint"):
        load_bundle(output)


def test_smote_retains_fractional_one_hot_features():
    config = load_experiment(BACKEND_ROOT / "configs/experiment.yaml")
    # Six fraud rows in distinct categories, all five other fraud rows are neighbors.
    X = np.zeros((26, len(FEATURE_COLUMNS)), dtype="uint8")
    for i in range(6):
        X[i, FEATURE_COLUMNS.index("type_TRANSFER" if i % 2 else "type_CASH_OUT")] = 1
    X[6:, FEATURE_COLUMNS.index("type_PAYMENT")] = 1
    y = np.array([1] * 6 + [0] * 20, dtype="uint8")
    original = X.copy()
    resampled, labels, audit = resample_training(X, y, config)
    np.testing.assert_array_equal(X, original)
    np.testing.assert_array_equal(resampled[:26], X)
    assert resampled.dtype.kind == "f"
    assert audit["multiple_positive_type_rows"] > 0
    assert audit["fractional_indicator_counts"]["type_TRANSFER"] > 0
    assert np.all(labels[26:] == 1)
    assert len(labels) == 40


@pytest.mark.parametrize("labels,message", [
    ([0] * 10 + [1] * 5, "Too few"),
    ([0] * 6 + [1] * 7, "minority"),
    ([0, 1, 2], "binary"),
])
def test_invalid_training_labels(labels, message):
    with pytest.raises(ValueError, match=message):
        validate_labels(labels, len(labels), 5)


def test_mismatched_preparation_rejected(prepared, tmp_path):
    path, config = prepared
    config.seeds.split = 12
    with pytest.raises(ValueError, match="does not match"):
        train_models(path, config, tmp_path / "models")
    assert not (tmp_path / "models").exists()


def test_failed_training_cannot_be_loaded(prepared, tmp_path, monkeypatch):
    path, config = prepared
    def fail(*args, **kwargs):
        raise RuntimeError("synthetic failure")
    monkeypatch.setattr("integritree.ml.training.RandomForestClassifier.fit", fail)
    with pytest.raises(RuntimeError):
        train_models(path, config, tmp_path / "models", "failed", progress=lambda _: None)
    assert json.loads((tmp_path / "models/failed/metadata.json").read_text())["status"] == "failed"
    with pytest.raises(ValueError, match="incomplete"):
        load_bundle(tmp_path / "models/failed")


def test_cli_trains_from_explicit_paths(prepared, tmp_path, monkeypatch, capsys):
    path, config = prepared
    config_path = tmp_path / "experiment.yaml"
    import yaml
    config_path.write_text(yaml.safe_dump(config.model_dump()))
    monkeypatch.setenv("INTEGRITREE_ARTIFACTS_DIR", str(tmp_path / "models"))
    monkeypatch.setattr("sys.argv", ["integritree-train", "--prepared", str(path),
                                     "--config", str(config_path), "--run-id", "cli"])
    assert main() == 0
    assert json.loads(capsys.readouterr().out)["success"]
