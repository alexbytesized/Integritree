"""Batched source audit, exact duplicates, identity preservation, and shared splits."""

import numpy as np
import pandas as pd
import pytest

from integritree.ml.data import (
    RAW_COLUMNS, audit_source, file_sha256, stratified_membership,
)
from integritree.ml.features import DataValidationError
from integritree.contracts import DatasetIdentity


@pytest.fixture
def synthetic_source(tmp_path, raw_record):
    rows = []
    for i in range(200):
        row = raw_record | {"nameOrig": f"C_SOURCE_{i}", "step": i + 1,
                            "amount": float(i), "isFraud": int(i % 5 == 0)}
        rows.append(row)
    frame = pd.DataFrame(rows).loc[:, RAW_COLUMNS]
    path = tmp_path / "synthetic.csv"
    frame.to_csv(path, index=False)
    return path, frame


def identity(path, rows):
    return DatasetIdentity(filename=path.name, source_url="https://example.com/synthetic",
                           sha256=file_sha256(path), row_count=rows)


def test_exact_duplicates_and_different_records_with_same_features(synthetic_source, tmp_path, monkeypatch):
    monkeypatch.setattr("integritree.ml.data.CSV_BLOCK_BYTES", 2048)
    path, frame = synthetic_source
    other_identity = frame.iloc[[0]].copy()
    other_identity["nameOrig"] = "C_ANOTHER"
    combined = pd.concat([frame, frame.iloc[[0]], other_identity], ignore_index=True)
    combined.to_csv(path, index=False)
    original_hash = file_sha256(path)
    output = tmp_path / "audit"
    output.mkdir()
    audit = audit_source(path, identity(path, 202), output, progress=lambda _: None)
    assert audit["duplicate_rows"] == 1
    assert audit["retained_rows"] == 201
    assert audit["zero_amounts"] == 3
    assert audit["missing"]["amount"] == 0
    assert audit["class_counts"] == {"0": 160, "1": 42}
    duplicates = pd.read_parquet(output / "duplicates.parquet")
    assert duplicates.to_dict("records") == [{"source_row_number": 201, "first_source_row_number": 1}]
    source = pd.read_parquet(output / "source.parquet")
    assert 202 in source["source_row_number"].values
    assert file_sha256(path) == original_hash


@pytest.mark.parametrize("column,value", [
    ("isFraud", ""), ("isFraud", 2), ("step", ""), ("nameDest", ""),
    ("type", "UNKNOWN"), ("oldbalanceOrg", -10),
])
def test_invalid_source_stops_with_audit(synthetic_source, tmp_path, column, value):
    import json
    path, frame = synthetic_source
    frame = frame.astype(object)
    frame.loc[0, column] = value
    frame.to_csv(path, index=False)
    output = tmp_path / "audit"
    output.mkdir()
    with pytest.raises(DataValidationError):
        audit_source(path, identity(path, len(frame)), output, progress=lambda _: None)
    audit = json.loads((output / "audit.json").read_text())
    assert audit["status"] == "failed"
    assert column in audit["issues"]


def test_missing_amount_type_allowed_and_reported(synthetic_source, tmp_path):
    path, frame = synthetic_source
    frame = frame.astype(object)
    frame.loc[0, "amount"] = None
    frame.loc[1, "type"] = None
    frame.to_csv(path, index=False)
    output = tmp_path / "audit"
    output.mkdir()
    audit = audit_source(path, identity(path, len(frame)), output, progress=lambda _: None)
    assert audit["missing"]["amount"] == audit["missing"]["type"] == 1
    source = pd.read_parquet(output / "source.parquet")
    assert np.isnan(source.loc[0, "amount"])
    assert source.loc[1, "type"] == "unknown"


def test_wrong_fingerprint_rejected(synthetic_source, tmp_path):
    path, frame = synthetic_source
    expected = identity(path, len(frame)).model_copy(update={"sha256": "0" * 64})
    output = tmp_path / "audit"
    output.mkdir()
    with pytest.raises(ValueError, match="SHA-256"):
        audit_source(path, expected, output, progress=lambda _: None)
    assert not (output / "source.parquet").exists()


def test_stratified_disjoint_reproducible_and_all_records_assigned():
    labels = np.tile([0, 0, 0, 0, 1], 40).astype("uint8")
    first = stratified_membership(labels, 42)
    np.testing.assert_array_equal(first, stratified_membership(labels, 42))
    assert np.bincount(first).tolist() == [160, 20, 20]
    for code, expected_fraud in enumerate([32, 4, 4]):
        assert int(labels[first == code].sum()) == expected_fraud
    assert set(np.unique(first)) == {0, 1, 2}


@pytest.mark.parametrize("labels", [[0] * 10, [0, 1], [0] * 20 + [1]])
def test_tiny_or_single_class_data_fails(labels):
    with pytest.raises(ValueError):
        stratified_membership(np.array(labels, dtype="uint8"), 42)


def test_malformed_csv_width_rejected(synthetic_source, tmp_path):
    import pyarrow as pa
    path, frame = synthetic_source
    with path.open("a") as handle:
        handle.write("too,few,columns\n")
    output = tmp_path / "audit"
    output.mkdir()
    with pytest.raises(pa.ArrowInvalid):
        audit_source(path, identity(path, len(frame) + 1), output, progress=lambda _: None)
