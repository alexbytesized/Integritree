"""Execute Phase 2 and publish a complete, traceable preparation bundle."""

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import re
import sys
import uuid

import numpy as np
import pandas as pd
import pyarrow as pa

from integritree.config import ExperimentConfig, load_experiment
from integritree.ml.data import (
    SPLIT_NAMES, ParquetSink, audit_source, file_sha256, parquet_batches,
    stratified_membership, write_json,
)
from integritree.ml.features import FEATURE_COLUMNS, SOURCE_COLUMNS, TIME_CONVENTION, DataValidationError
from integritree.ml.preprocessing import FittedPreprocessor
from integritree.settings import load_settings


def validate_preparation_config(config: ExperimentConfig) -> None:
    config.require_ready("prepare")
    expected = {
        "features": FEATURE_COLUMNS, "time_convention": TIME_CONVENTION,
        "normalization": "minmax", "missing_values": "median_unknown", "duplicates": "drop_exact",
    }
    for name, value in expected.items():
        if getattr(config.preprocessing, name) != value:
            raise ValueError(f"Unsupported preparation setting preprocessing.{name}; use the approved Phase 2 policy")


def selected_batches(source: Path, membership: np.ndarray, split: int):
    offset = 0
    for frame in parquet_batches(source, columns=SOURCE_COLUMNS):
        selected = membership[offset:offset + len(frame)] == split
        offset += len(frame)
        if selected.any():
            yield frame.loc[selected, SOURCE_COLUMNS]


def prepare_dataset(source: Path, config: ExperimentConfig, output_root: Path,
                    run_id: str | None = None, progress=print) -> Path:
    validate_preparation_config(config)
    run_id = run_id or ("prepare_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8])
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,100}", run_id):
        raise ValueError("run_id must contain only letters, digits, underscores, or hyphens")
    output = output_root.resolve() / run_id
    # Never overwrite an existing completed or failed run.
    output.mkdir(parents=True, exist_ok=False)
    metadata = {"schema_version": 1, "status": "running", "run_id": run_id,
                "created_at": datetime.now(timezone.utc).isoformat(), "dataset": config.dataset.model_dump()}
    write_json(output / "metadata.json", metadata)
    sinks = []
    try:
        write_json(output / "configuration.json", config.model_dump())
        progress("Verifying source fingerprint and auditing records in batches")
        audit = audit_source(source, config.dataset, output, progress)
        n = audit["retained_rows"]
        labels = np.memmap(output / "_labels.bin", mode="r", dtype="uint8", shape=(n,))
        progress("Creating shared stratified 80/10/10 split membership")
        membership = stratified_membership(labels, config.seeds.split)
        split_counts = {}
        for code, name in enumerate(SPLIT_NAMES):
            counts = np.bincount(labels[membership == code], minlength=2)
            split_counts[name] = {"rows": int(counts.sum()), "legitimate": int(counts[0]),
                                  "fraud": int(counts[1]), "fraud_fraction": float(counts[1] / counts.sum())}
        del labels

        training_rows = split_counts["train"]["rows"]
        amounts = np.memmap(output / "_training_amounts.bin", mode="w+", dtype="float64", shape=(training_rows,))
        offset = 0
        for frame in selected_batches(output / "source.parquet", membership, 0):
            end = offset + len(frame)
            amounts[offset:end] = frame["amount"].to_numpy(dtype="float64", na_value=np.nan)
            offset = end
        if offset != training_rows:
            raise ValueError("Training record alignment failed")
        if np.isnan(amounts).all():
            raise ValueError("Training amount column is entirely missing")
        # Scratch values may be rearranged to compute the median; source order is preserved elsewhere.
        median = float(np.nanmedian(amounts, overwrite_input=True))
        del amounts
        progress(f"Fitting preprocessing on {training_rows:,} original training records only")
        preprocessor = FittedPreprocessor.fit_batches(
            median, selected_batches(output / "source.parquet", membership, 0))
        preprocessor.save(output / "preprocessing.json")

        manifest = ParquetSink(output / "split_manifest.parquet")
        feature_sinks = {name: ParquetSink(output / f"{name}_features.parquet") for name in SPLIT_NAMES}
        sinks = [manifest, *feature_sinks.values()]
        offset = 0
        for frame in parquet_batches(output / "source.parquet"):
            codes = membership[offset:offset + len(frame)]
            offset += len(frame)
            manifest.write(pd.DataFrame({
                "source_row_number": frame["source_row_number"].to_numpy(dtype="int64"),
                "split": pd.Series([SPLIT_NAMES[c] for c in codes], dtype="string[pyarrow]"),
                "actual_label": frame["isFraud"].to_numpy(dtype="uint8"),
            }))
            for code, name in enumerate(SPLIT_NAMES):
                selected = codes == code
                if not selected.any():
                    continue
                features = preprocessor.transform(frame.loc[selected, SOURCE_COLUMNS])
                features.insert(0, "source_row_number", frame.loc[selected, "source_row_number"].to_numpy(dtype="int64"))
                feature_sinks[name].write(features)
            if offset // 500000 != (offset - len(frame)) // 500000:
                progress(f"Transformed {offset:,} retained rows")
        for sink in sinks:
            sink.close()
        sinks = []
        if offset != n:
            raise ValueError("Prepared record alignment failed")
        metadata.update({
            "status": "complete", "retained_rows": n, "removed_duplicate_rows": audit["duplicate_rows"],
            "feature_order": FEATURE_COLUMNS, "splits": split_counts,
            "split_algorithm": "sklearn train_test_split: stratified 80/20, then 50/50 held-out; same seed for both calls",
            "split_seed": config.seeds.split,
            "transaction_identity": "<dataset.sha256>:<one-based original data-row number>",
            "configuration_sha256": file_sha256(output / "configuration.json"),
            "split_manifest_sha256": file_sha256(output / "split_manifest.parquet"),
            "package_versions": {p: version(p) for p in ["integritree-backend", "numpy", "pandas", "pyarrow", "scikit-learn"]},
            "implementation_sha256": {p.name: file_sha256(p) for p in [
                Path(__file__), Path(__file__).with_name("data.py"),
                Path(__file__).with_name("features.py"), Path(__file__).with_name("preprocessing.py"),
            ]},
        })
        metadata["files"] = {p.name: file_sha256(p) for p in output.iterdir()
                             if p.is_file() and p.name != "metadata.json" and not p.name.startswith("_")}
        write_json(output / "metadata.json", metadata)
        progress(f"Preparation complete: {n:,} retained rows, {audit['duplicate_rows']:,} exact duplicates removed")
        return output
    except Exception as exc:
        metadata.update(status="failed", failure_type=type(exc).__name__)
        write_json(output / "metadata.json", metadata)
        raise
    finally:
        for sink in sinks:
            sink.close()
        # Only known scratch files within this run; never recursive deletion.
        for name in ("_duplicates.sqlite", "_duplicates.sqlite-journal", "_labels.bin", "_training_amounts.bin"):
            scratch = output / name
            if scratch.is_file():
                try:
                    scratch.unlink()
                except PermissionError:
                    # A Windows memory map may remain open during exception unwinding.
                    pass


def load_prepared_split(bundle: Path, split: str):
    """Load aligned X, y, source-row IDs. Requires a completed, intact bundle.

    This convenience reader materializes one split; preparation itself is batched.
    The source SHA in metadata completes each source-row transaction identity.
    """
    if split not in SPLIT_NAMES:
        raise ValueError("Unknown split")
    metadata = json.loads((bundle / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("status") != "complete" or metadata.get("schema_version") != 1:
        raise ValueError("Preparation bundle is incomplete or unsupported")
    if metadata["feature_order"] != FEATURE_COLUMNS:
        raise ValueError("Unsupported feature order")
    for name in ("split_manifest.parquet", f"{split}_features.parquet"):
        if file_sha256(bundle / name) != metadata["files"][name]:
            raise ValueError(f"Preparation file fingerprint mismatch: {name}")
    features = pd.read_parquet(bundle / f"{split}_features.parquet")
    manifest = pd.read_parquet(bundle / "split_manifest.parquet", filters=[("split", "==", split)])
    if not np.array_equal(features["source_row_number"], manifest["source_row_number"]):
        raise ValueError("Features and labels are not aligned")
    if len(features) != metadata["splits"][split]["rows"] or not features["source_row_number"].is_unique:
        raise ValueError("Invalid split identities or row count")
    return (features.loc[:, FEATURE_COLUMNS], manifest["actual_label"],
            features["source_row_number"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Absolute path or relative to backend root")
    parser.add_argument("--run-id", help="Optional unique preparation run name")
    args = parser.parse_args()
    try:
        settings = load_settings()
        config_path = args.config or settings.experiment_config
        if not config_path.is_absolute():
            config_path = settings.backend_root / config_path
        config = load_experiment(config_path)
        output = prepare_dataset(settings.dataset_path, config, settings.prepared_dir,
                                 args.run_id, progress=lambda message: print(message, file=sys.stderr, flush=True))
    except (ValueError, OSError, pa.ArrowException) as exc:
        if isinstance(exc, pa.ArrowException):
            message = "CSV/Parquet parsing failed; check column counts and formats. Raw values are not logged."
        elif isinstance(exc, DataValidationError):
            message = str(exc) + "; see audit.json for counts and row positions"
        else:
            message = str(exc)
        print(json.dumps({"success": False, "error": message}), file=sys.stderr)
        return 2
    print(json.dumps({"success": True, "bundle": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
