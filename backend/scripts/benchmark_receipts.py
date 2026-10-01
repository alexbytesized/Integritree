"""Offline receipt audit/OCR benchmark, independent of the ML environment.

Run with the isolated OCR Python. Sensitive outputs belong under runtime/receipt_checks.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import statistics
import sys
from time import perf_counter

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND / "src"))
os.environ.setdefault("OMP_THREAD_LIMIT", "2")

from integritree.receipts.audit import file_hash, inventory, split_inventory
from integritree.receipts.gcash import parse_candidates
from integritree.receipts.ocr import RapidEngine, TesseractEngine, extract

FIELDS = ("workflow", "category", "amount", "date", "time", "reference")


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def code_hashes():
    files = [Path(__file__), *(BACKEND / "src/integritree/receipts").glob("*.py")]
    return {p.relative_to(BACKEND).as_posix(): file_hash(p) for p in sorted(files)}


def no_network(*args, **kwargs):
    raise RuntimeError("Network access disabled during receipt processing")


def model_hashes(tessdata):
    import rapidocr

    weights = Path(rapidocr.__file__).parent / "models"
    paths = list(weights.glob("*.onnx")) + [Path(tessdata) / "eng.traineddata"]
    if not paths[-1].is_file():
        raise ValueError("Missing local eng.traineddata")
    return {p.name: file_hash(p) for p in sorted(paths)}


def checked_path(root, row):
    path = (root / row["path"]).resolve()
    if not path.is_relative_to(root.resolve()) or file_hash(path) != row["sha256"]:
        raise ValueError("Sample path/hash does not match frozen inventory")
    return path


def run(args):
    audit = read_json(args.audit)
    rows = [r for r in audit["rows"] if r["split"] == args.split]
    if not rows:
        raise ValueError("No eligible samples in requested split")
    root = Path(args.samples).resolve()
    for row in rows:
        checked_path(root, row)
    hashes = code_hashes()
    weights = model_hashes(args.tessdata)
    if args.split == "verification":
        if not args.freeze_from:
            raise ValueError(
                "Verification requires --freeze-from completed development run"
            )
        frozen = read_json(Path(args.freeze_from) / "run.json")
        if (
            frozen["split"] != "development"
            or not (Path(args.freeze_from) / "complete.json").exists()
        ):
            raise ValueError("Freeze source is not a completed development run")
        if (
            frozen["code_sha256"] != hashes
            or frozen["model_sha256"] != weights
            or frozen["audit_sha256"] != file_hash(args.audit)
        ):
            raise ValueError(
                "Code, weights, or sample split changed since development freeze"
            )
    out = Path(args.output).resolve()
    if out == root or out.is_relative_to(root):
        raise ValueError("Benchmark output must not be inside source samples")
    out.mkdir(parents=True, exist_ok=False)
    versions = {
        key: importlib.metadata.version(key)
        for key in (
            "rapidocr",
            "onnxruntime",
            "tesserocr",
            "Pillow",
            "numpy",
            "opencv-python",
            "omegaconf",
        )
    }
    import tesserocr

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "split": args.split,
        "count": len(rows),
        "audit_sha256": file_hash(args.audit),
        "code_sha256": hashes,
        "model_sha256": weights,
        "versions": versions,
        "tesseract_version": tesserocr.tesseract_version(),
        "tesseract_psm": 11,
        "tesseract_oem": "LSTM_ONLY",
        "tesseract_language": "eng",
        "input_policy": "original RGB image, no engine-specific cropping or tuning",
        "freeze_from": str(Path(args.freeze_from).resolve())
        if args.freeze_from
        else None,
        "network": "socket connections disabled for all receipt processing",
    }
    write_json(out / "run.json", metadata)
    # All weights must be provisioned in setup. A missing model cannot trigger a download.
    socket.socket.connect = no_network
    socket.socket.connect_ex = no_network
    socket.create_connection = no_network
    summary = {}
    for name, factory in (
        ("rapidocr", RapidEngine),
        ("tesseract", lambda: TesseractEngine(args.tessdata)),
    ):
        start = perf_counter()
        engine = factory()
        initialization = perf_counter() - start
        times, errors, layouts = [], 0, Counter()
        try:
            with (out / (name + ".jsonl")).open("x", encoding="utf-8") as stream:
                for index, row in enumerate(rows, 1):
                    result = {
                        "path": row["path"],
                        "sha256": row["sha256"],
                        "split": row["split"],
                    }
                    try:
                        extracted = extract(engine, checked_path(root, row))
                        result.update(extracted)
                        result["parsed"] = parse_candidates(extracted["lines"])
                        result["status"] = "complete"
                        layouts[
                            result["parsed"]["fields"]["workflow"] or "unresolved"
                        ] += 1
                        times.append(result["seconds"])
                    except Exception as exc:
                        # No supplied text or raw exception message in console/log summary.
                        result.update(status="failed", error_type=type(exc).__name__)
                        errors += 1
                    stream.write(
                        json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n"
                    )
                    stream.flush()
                    if index % 10 == 0 or index == len(rows):
                        print(
                            f"{name}: {index}/{len(rows)} complete; failures={errors}",
                            flush=True,
                        )
        finally:
            engine.close()
        summary[name] = {
            "initialization_seconds": initialization,
            "images": len(rows),
            "failed": errors,
            "median_seconds": statistics.median(times) if times else None,
            "p95_seconds": sorted(times)[min(len(times) - 1, int(0.95 * len(times)))]
            if times
            else None,
            "provisional_workflows": dict(layouts),
        }
    write_json(out / "complete.json", summary)
    print(json.dumps(summary, indent=2))


def score_records(results, annotations):
    """Null expected fields are visibly unavailable, not an excuse to hallucinate."""
    lookup = {row["path"]: row for row in results}
    counters = {field: Counter() for field in FIELDS}
    all_required, failures = 0, 0
    for annotation in annotations:
        row = lookup.get(annotation["path"])
        if row is None or row["sha256"] != annotation["sha256"]:
            raise ValueError("Missing benchmark row or annotation fingerprint mismatch")
        actual = row.get("parsed", {}).get("fields", {})
        failed = row["status"] != "complete"
        failures += failed
        required_ok = not failed
        for field in FIELDS:
            expected = annotation["fields"][field]
            observed = actual.get(field)
            if expected is None:
                status = (
                    "unexpected" if observed is not None else "correctly_unavailable"
                )
            elif observed is None:
                status = "missing"
            elif expected == observed:
                status = "correct"
            else:
                status = "wrong"
            counters[field][status] += 1
            if field != "reference" and status not in {
                "correct",
                "correctly_unavailable",
            }:
                required_ok = False
        all_required += required_ok
    return {
        "annotated_images": len(annotations),
        "engine_failures": failures,
        "all_required_fields_match": all_required,
        "fields": {k: dict(v) for k, v in counters.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    a = commands.add_parser("audit")
    a.add_argument("--samples", type=Path, required=True)
    a.add_argument("--output", type=Path, required=True)
    r = commands.add_parser("run")
    for key in ("samples", "audit", "output", "tessdata"):
        r.add_argument("--" + key, type=Path, required=True)
    r.add_argument("--split", choices=("development", "verification"), required=True)
    r.add_argument("--freeze-from", type=Path)
    s = commands.add_parser("score")
    s.add_argument("--run", type=Path, required=True)
    s.add_argument("--annotations", type=Path, required=True)
    s.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "audit":
        report = inventory(args.samples)
        report["rows"] = split_inventory(report)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, report)
        print(
            json.dumps(
                {
                    "counts": report["counts"],
                    "status": report["status_counts"],
                    "duplicate_groups": len(report["decoded_duplicate_groups"]),
                    "splits": dict(Counter(r["split"] for r in report["rows"])),
                },
                indent=2,
            )
        )
    elif args.command == "run":
        run(args)
    else:
        annotations = read_json(args.annotations)
        if annotations.get("annotation_method") != "visual_inspection_of_originals":
            raise ValueError("Require visual labels independent of OCR output")
        metadata = read_json(args.run / "run.json")
        if not (args.run / "complete.json").exists():
            raise ValueError("Cannot score incomplete benchmark")
        selected = [r for r in annotations["rows"] if r["split"] == metadata["split"]]
        if not selected or len({r["path"] for r in selected}) != len(selected):
            raise ValueError("Empty or duplicate annotation rows")
        result = {
            "run_sha256": file_hash(args.run / "run.json"),
            "annotations_sha256": file_hash(args.annotations),
            "split": metadata["split"],
            "coverage": f"{len(selected)} visually annotated images of {metadata['count']} processed",
        }
        for name in ("rapidocr", "tesseract"):
            rows = [
                json.loads(line)
                for line in (args.run / (name + ".jsonl"))
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            result[name] = score_records(rows, selected)
        write_json(args.output, result)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
