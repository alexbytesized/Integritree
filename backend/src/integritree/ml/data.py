"""Batched source validation, disk-backed exact deduplication, and shared splits."""

import csv
import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split

from integritree.contracts import DatasetIdentity
from integritree.ml.features import (
    SOURCE_COLUMNS, TRANSACTION_TYPES, DataValidationError, validate_predictors,
)

RAW_COLUMNS = [
    "step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
    "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud",
]
BALANCE_COLUMNS = ["oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
SPLIT_NAMES = ("train", "validation", "test")
BATCH_ROWS = 25000
CSV_BLOCK_BYTES = 2**20


def file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def parquet_batches(path: Path, columns=None):
    for batch in pq.ParquetFile(path).iter_batches(batch_size=BATCH_ROWS, columns=columns):
        yield batch.to_pandas(types_mapper=pd.ArrowDtype)


def validate_raw(frame: pd.DataFrame) -> pd.DataFrame:
    if list(frame.columns) != RAW_COLUMNS:
        raise ValueError("PaySim CSV must contain the eleven expected columns in source order")
    # Unknown is reserved for missing categories, not silently accepted source typos.
    types = frame["type"].astype("string[pyarrow]").str.strip()
    bad_types = types.notna() & types.ne("") & ~types.isin(TRANSACTION_TYPES)
    if bad_types.any():
        raise DataValidationError({"type": {"count": int(bad_types.sum()),
                                          "row_positions": np.flatnonzero(bad_types)[:20].tolist()}})
    result = frame.copy()
    predictors = validate_predictors(frame[SOURCE_COLUMNS])
    for name in SOURCE_COLUMNS:
        result[name] = predictors[name]
    issues = {}
    for name in BALANCE_COLUMNS + ["isFraud", "isFlaggedFraud"]:
        text = frame[name].astype("string[pyarrow]").str.strip()
        missing = text.isna() | text.eq("").fillna(False)
        values = pd.to_numeric(text, errors="coerce").to_numpy(dtype="float64", na_value=np.nan)
        bad = ~missing.to_numpy() & (~np.isfinite(values) | (values < 0))
        if name in ("isFraud", "isFlaggedFraud"):
            bad |= ~missing.to_numpy() & ~np.isin(values, [0, 1])
        if name == "isFraud":
            bad |= missing.to_numpy()
        if bad.any():
            issues[name] = {"count": int(bad.sum()), "row_positions": np.flatnonzero(bad)[:20].tolist()}
        result[name] = values
    if issues:
        raise DataValidationError(issues)
    result["isFraud"] = result["isFraud"].astype("uint8")
    return result.loc[:, RAW_COLUMNS]


class ParquetSink:
    def __init__(self, path: Path):
        self.path = path
        self.writer = None

    def write(self, frame: pd.DataFrame):
        table = pa.Table.from_pandas(frame, preserve_index=False)
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.path, table.schema, compression="zstd")
        self.writer.write_table(table)

    def close(self):
        if self.writer is not None:
            self.writer.close()


def audit_source(path: Path, identity: DatasetIdentity, output: Path, progress=print) -> dict:
    """Write retained source/removed-row IDs; fail with a saved incomplete audit."""
    audit = {"status": "running", "source": identity.model_dump(), "scanned_rows": 0,
             "retained_rows": 0, "duplicate_rows": 0, "missing": dict.fromkeys(RAW_COLUMNS, 0),
             "class_counts": {}, "retained_class_counts": {}, "type_counts": {},
             "zero_amounts": 0, "numeric_ranges": {}, "duplicate_comparison": "all 11 typed original columns"}
    source_sink = ParquetSink(output / "source.parquet")
    duplicate_sink = ParquetSink(output / "duplicates.parquet")
    connection = None
    try:
        fingerprint = file_sha256(path)
        audit["observed_sha256"] = fingerprint
        if fingerprint != identity.sha256:
            raise ValueError("Dataset SHA-256 differs from the approved configuration")
        with path.open(encoding="utf-8-sig", newline="") as handle:
            if next(csv.reader(handle), None) != RAW_COLUMNS:
                raise ValueError("CSV header does not match the eleven expected source columns")
        connection = sqlite3.connect(output / "_duplicates.sqlite")
        connection.execute("PRAGMA cache_size=-8192")
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("CREATE TABLE seen (record TEXT PRIMARY KEY, first_row INTEGER NOT NULL) WITHOUT ROWID")
        reader = pacsv.open_csv(
            path, read_options=pacsv.ReadOptions(block_size=CSV_BLOCK_BYTES),
            parse_options=pacsv.ParseOptions(ignore_empty_lines=False),
            convert_options=pacsv.ConvertOptions(column_types={n: pa.string() for n in RAW_COLUMNS},
                                                strings_can_be_null=True, null_values=[""]),
        )
        with (output / "_labels.bin").open("wb") as labels_file:
            for batch in reader:
                frame = batch.to_pandas(types_mapper=pd.ArrowDtype)
                start = audit["scanned_rows"] + 1
                for name in RAW_COLUMNS:
                    text = frame[name].astype("string[pyarrow]").str.strip()
                    audit["missing"][name] += int((text.isna() | text.eq("").fillna(False)).sum())
                try:
                    clean = validate_raw(frame)
                except DataValidationError as exc:
                    audit["invalid_batch_first_source_row"] = start
                    audit["issues"] = exc.issues
                    raise
                audit["scanned_rows"] += len(clean)
                for key, column in [("class_counts", "isFraud"), ("type_counts", "type")]:
                    counts = Counter(audit[key])
                    counts.update({str(k): int(v) for k, v in clean[column].value_counts().items()})
                    audit[key] = dict(counts)
                audit["zero_amounts"] += int(clean["amount"].eq(0).sum())
                for name in ["step", "amount", *BALANCE_COLUMNS]:
                    observed = clean[name].dropna()
                    if not observed.empty:
                        low, high = float(observed.min()), float(observed.max())
                        prior = audit["numeric_ranges"].get(name, {"min": low, "max": high})
                        audit["numeric_ranges"][name] = {"min": min(prior["min"], low), "max": max(prior["max"], high)}
                keep, removed, first_rows = [], [], []
                # Full canonical keys, not hashes: collisions cannot discard records.
                for position, row in enumerate(clean.itertuples(index=False, name=None)):
                    key = json.dumps([None if pd.isna(v) else (0 if v == 0 else v) for v in row],
                                     separators=(",", ":"), allow_nan=False)
                    source_row = start + position
                    cursor = connection.execute("INSERT OR IGNORE INTO seen VALUES (?, ?)", (key, source_row))
                    if cursor.rowcount:
                        keep.append(position)
                    else:
                        removed.append(source_row)
                        first_rows.append(connection.execute("SELECT first_row FROM seen WHERE record=?", (key,)).fetchone()[0])
                connection.commit()
                clean.insert(0, "source_row_number", np.arange(start, start + len(clean), dtype="int64"))
                retained = clean.iloc[keep]
                if len(retained):
                    source_sink.write(retained)
                    retained["isFraud"].to_numpy(dtype="uint8").tofile(labels_file)
                    counts = Counter(audit["retained_class_counts"])
                    counts.update({str(k): int(v) for k, v in retained["isFraud"].value_counts().items()})
                    audit["retained_class_counts"] = dict(counts)
                if removed:
                    duplicate_sink.write(pd.DataFrame({"source_row_number": removed, "first_source_row_number": first_rows}, dtype="int64"))
                audit["retained_rows"] += len(retained)
                audit["duplicate_rows"] += len(removed)
                if audit["scanned_rows"] // 500000 != (start - 1) // 500000:
                    progress(f"Audited {audit['scanned_rows']:,} source rows; exact duplicates: {audit['duplicate_rows']:,}")
        if audit["scanned_rows"] != identity.row_count:
            raise ValueError("Dataset row count differs from the approved configuration")
        if not audit["retained_rows"]:
            raise ValueError("Dataset has no retained records")
        if file_sha256(path) != fingerprint:
            raise ValueError("Source file changed during preparation audit")
        if duplicate_sink.writer is None:
            duplicate_sink.write(pd.DataFrame({"source_row_number": pd.Series(dtype="int64"),
                                               "first_source_row_number": pd.Series(dtype="int64")}))
        audit["status"] = "complete"
        return audit
    except Exception as exc:
        audit["status"] = "failed"
        # Parser errors can include raw contents: never persist arbitrary exception text.
        audit["failure_type"] = type(exc).__name__
        raise
    finally:
        source_sink.close()
        duplicate_sink.close()
        if connection is not None:
            connection.close()
        write_json(output / "audit.json", audit)


def stratified_membership(labels: np.ndarray, seed: int) -> np.ndarray:
    """80% train, then split the held-out 20% equally; 0/1/2 encode split names."""
    if set(np.unique(labels).tolist()) != {0, 1}:
        raise ValueError("Stratified preparation requires both legitimate and fraud labels")
    positions = np.arange(len(labels), dtype="int64")
    try:
        train, held = train_test_split(positions, test_size=0.2, random_state=seed, stratify=labels)
        validation, test = train_test_split(held, test_size=0.5, random_state=seed, stratify=labels[held])
    except ValueError as exc:
        raise ValueError("Dataset is too small for the requested stratified 80/10/10 split") from exc
    for indices in (train, validation, test):
        if len(np.unique(labels[indices])) != 2:
            raise ValueError("Each split must contain both classes; dataset is too small")
    membership = np.empty(len(labels), dtype="uint8")
    membership[train], membership[validation], membership[test] = 0, 1, 2
    return membership
