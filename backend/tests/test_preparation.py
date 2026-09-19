"""End-to-end preparation on tiny synthetic records, without real dataset access."""

import json

import numpy as np
import pandas as pd
import pytest

from integritree.config import load_experiment
from integritree.settings import BACKEND_ROOT
from integritree.ml.data import RAW_COLUMNS, file_sha256
from integritree.ml.features import FEATURE_COLUMNS, SOURCE_COLUMNS
from integritree.ml.preparation import prepare_dataset, load_prepared_split
from integritree.ml.preprocessing import FittedPreprocessor


@pytest.fixture
def source_config(tmp_path, raw_record):
    frame = pd.DataFrame([raw_record | {
        "nameOrig": f"C_{i}", "step": i + 1, "amount": float(i),
        "isFraud": int(i % 5 == 0),
    } for i in range(200)]).loc[:, RAW_COLUMNS]
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    path = tmp_path / "synthetic.csv"
    frame.to_csv(path, index=False)
    config = load_experiment(BACKEND_ROOT / "configs/experiment.yaml").model_copy(deep=True)
    config.dataset = config.dataset.model_copy(update={
        "filename": path.name, "sha256": file_sha256(path), "row_count": len(frame),
    })
    return path, config


def test_end_to_end_bundle_alignment_reload_and_reproducibility(source_config, tmp_path):
    source, config = source_config
    output = prepare_dataset(source, config, tmp_path / "prepared", "first", progress=lambda _: None)
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["status"] == "complete"
    assert metadata["removed_duplicate_rows"] == 1
    assert not list(output.glob("_*"))
    retained = pd.read_parquet(output / "source.parquet").set_index("source_row_number")
    preprocessor = FittedPreprocessor.load(output / "preprocessing.json")
    identities = []
    for split, size in [("train", 160), ("validation", 20), ("test", 20)]:
        X, y, ids = load_prepared_split(output, split)
        assert list(X.columns) == FEATURE_COLUMNS
        assert len(X) == len(y) == len(ids) == size
        np.testing.assert_array_equal(y, retained.loc[ids, "isFraud"])
        expected = preprocessor.transform(retained.loc[ids, SOURCE_COLUMNS])
        np.testing.assert_allclose(X, expected)
        identities.extend(ids.tolist())
        if split == "train":
            assert preprocessor.state.amount_median == retained.loc[ids, "amount"].median()
    assert len(set(identities)) == 200
    assert 201 not in identities
    original_metadata = (output / "metadata.json").read_bytes()
    with pytest.raises(FileExistsError):
        prepare_dataset(source, config, tmp_path / "prepared", "first", progress=lambda _: None)
    assert (output / "metadata.json").read_bytes() == original_metadata
    second = prepare_dataset(source, config, tmp_path / "prepared", "second", progress=lambda _: None)
    assert (output / "preprocessing.json").read_bytes() == (second / "preprocessing.json").read_bytes()
    pd.testing.assert_frame_equal(pd.read_parquet(output / "split_manifest.parquet"),
                                  pd.read_parquet(second / "split_manifest.parquet"))
    with (output / "train_features.parquet").open("ab") as handle:
        handle.write(b"modified")
    with pytest.raises(ValueError, match="fingerprint"):
        load_prepared_split(output, "train")


def test_failure_never_publishes_complete_bundle(source_config, tmp_path):
    source, config = source_config
    config.dataset = config.dataset.model_copy(update={"row_count": 202})
    with pytest.raises(ValueError, match="row count"):
        prepare_dataset(source, config, tmp_path / "prepared", "failed", progress=lambda _: None)
    metadata = json.loads((tmp_path / "prepared/failed/metadata.json").read_text())
    assert metadata["status"] == "failed"
    with pytest.raises(ValueError, match="incomplete"):
        load_prepared_split(tmp_path / "prepared/failed", "train")


def test_unsupported_policy_rejected_before_creating_outputs(source_config, tmp_path):
    source, config = source_config
    config.preprocessing.normalization = "standard"
    with pytest.raises(ValueError, match="normalization"):
        prepare_dataset(source, config, tmp_path / "prepared", "unsupported")
    assert not (tmp_path / "prepared").exists()


def test_held_out_values_do_not_change_fitted_training_state(source_config, tmp_path):
    source, config = source_config
    first = prepare_dataset(source, config, tmp_path / "prepared", "baseline", progress=lambda _: None)
    manifest = pd.read_parquet(first / "split_manifest.parquet")
    held_id = int(manifest.loc[(manifest["split"] == "test") & (manifest["source_row_number"] > 1), "source_row_number"].iloc[0])
    frame = pd.read_csv(source)
    frame.loc[held_id - 1, "amount"] = 1e12
    frame.to_csv(source, index=False)
    config.dataset = config.dataset.model_copy(update={"sha256": file_sha256(source)})
    second = prepare_dataset(source, config, tmp_path / "prepared", "changed_held_out", progress=lambda _: None)
    assert (first / "preprocessing.json").read_bytes() == (second / "preprocessing.json").read_bytes()
    old_X, old_y, old_ids = load_prepared_split(first, "train")
    new_X, new_y, new_ids = load_prepared_split(second, "train")
    pd.testing.assert_frame_equal(old_X, new_X)
    np.testing.assert_array_equal(old_y, new_y)
    np.testing.assert_array_equal(old_ids, new_ids)
