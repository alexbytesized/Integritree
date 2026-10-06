"""Exercise the HTTP upload/export path using validation-derived and synthetic CSVs.

Does not read raw source records, the split manifest, or test features. Original
identifiers are unavailable in validation features: reconstruct equivalent five
predictors and obtain labels exclusively from the existing validation report.
"""

import argparse
import csv
import ctypes
from ctypes import wintypes
import io
import json
from pathlib import Path
import time
import zipfile
import httpx
import numpy as np
import pandas as pd
from integritree.ml.data import file_sha256
from integritree.ml.features import SOURCE_COLUMNS, TRANSACTION_TYPES
from integritree.ml.preprocessing import FittedPreprocessor
from integritree.settings import load_settings


def memory(pid):
    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
            (name, ctypes.c_size_t)
            for name in (
                "peak",
                "working",
                "peak_paged",
                "paged",
                "peak_nonpaged",
                "nonpaged",
                "pagefile",
                "peak_pagefile",
            )
        ]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(Counters),
        wintypes.DWORD,
    ]
    handle = kernel.OpenProcess(0x1000 | 0x10, False, pid)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return {
            "working_mib": counters.working / 2**20,
            "process_peak_mib": counters.peak / 2**20,
        }
    finally:
        kernel.CloseHandle(handle)


def fixtures(settings, root):
    prepared = settings.research_prepared_dir
    report = (
        settings.reports_dir / settings.research_model_dir.name / "final_validation"
    )
    metadata = json.loads((prepared / "metadata.json").read_text())
    assert (
        file_sha256(prepared / "validation_features.parquet")
        == metadata["files"]["validation_features.parquet"]
    )
    saved = pd.read_parquet(report / "predictions.parquet")
    assert (saved.split == "validation").all()
    features = pd.read_parquet(prepared / "validation_features.parquet")
    assert np.array_equal(features.source_row_number, saved.source_row_number)
    state = FittedPreprocessor.load(prepared / "preprocessing.json").state
    for name, scale, offset in zip(state.scaled_columns, state.scale, state.offset):
        features[name] = (features[name] - offset) / scale
    output = pd.DataFrame(
        {
            "step": (features.day_of_week * 24 + features.hour_of_day)
            .round()
            .astype(int)
            + 1,
            "type": np.array(TRANSACTION_TYPES)[
                features[[f"type_{t}" for t in TRANSACTION_TYPES]]
                .to_numpy()
                .argmax(axis=1)
            ],
            "amount": np.maximum(0, np.expm1(features.log_amount)),
            "nameOrig": np.where(
                features.is_merchant_origin == 1, "M_VALIDATION", "C_VALIDATION"
            ),
            "nameDest": np.where(
                features.is_merchant_dest == 1, "M_VALIDATION", "C_VALIDATION"
            ),
            "isFraud": saved.actual_label,
        }
    )
    validation = root / "validation_derived.csv"
    output.to_csv(validation, index=False)
    synthetic = root / "synthetic_million.csv"
    with synthetic.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(SOURCE_COLUMNS + ["isFraud"])
        writer.writerows(
            (
                i % 744 + 1,
                TRANSACTION_TYPES[i % 5],
                (i * 193) % 1000000 / 100,
                "C_SYNTHETIC",
                "M_SYNTHETIC" if i % 5 == 3 else "C_SYNTHETIC",
                int(i % 1000 == 0),
            )
            for i in range(1_000_000)
        )
    return validation, synthetic, saved


def run(client, path, pid, expected=None):
    token = client.post("/sessions").json()["token"]
    headers = {"X-Research-Session": token}
    started = time.monotonic()
    baseline = memory(pid)

    def chunks():
        with path.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                yield block

    response = client.post(
        "/analyses",
        params={"filename": path.name},
        headers=headers | {"Content-Type": "text/csv"},
        content=chunks(),
    )
    response.raise_for_status()
    identifier = response.json()["id"]
    peak = baseline["working_mib"]
    phases = []
    while True:
        response = client.get(f"/analyses/{identifier}", headers=headers)
        response.raise_for_status()
        job = response.json()
        peak = max(peak, memory(pid)["working_mib"])
        if not phases or phases[-1]["status"] != job["status"]:
            phases.append(
                {"status": job["status"], "seconds": time.monotonic() - started}
            )
            print(path.name, job["status"], job["rows_processed"], flush=True)
        if job["status"] in ("complete", "failed"):
            break
        time.sleep(0.5)
    assert job["status"] == "complete", job
    analysis_seconds = time.monotonic() - started
    client.post(f"/analyses/{identifier}/exports", headers=headers).raise_for_status()
    while True:
        export = client.get(f"/analyses/{identifier}", headers=headers).json()["export"]
        peak = max(peak, memory(pid)["working_mib"])
        if export["status"] in ("complete", "failed"):
            break
        time.sleep(0.5)
    assert export["status"] == "complete", export
    zipped = path.with_suffix(".zip")
    with client.stream(
        "GET", f"/analyses/{identifier}/exports/download", headers=headers
    ) as response:
        response.raise_for_status()
        with zipped.open("wb") as stream:
            for chunk in response.iter_bytes():
                stream.write(chunk)
    completed = time.monotonic()
    max_error = 0.0
    with zipfile.ZipFile(zipped) as archive:
        raw_name = next(n for n in archive.namelist() if n.startswith("Raw-Data_"))
        paper_name = raw_name.replace("Raw-Data_", "Experiment-Paper_").replace(
            ".csv", ".pdf"
        )
        assert sorted(archive.namelist()) == sorted([raw_name, paper_name])
        assert archive.read(paper_name).startswith(b"%PDF-")
        count = 0
        with archive.open(raw_name) as stream:
            for row in csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8")):
                if expected is not None:
                    for model in ("rf", "rf_smote"):
                        error = abs(
                            float(row[model + "_score"])
                            - expected[model + "_risk_score"].iat[count]
                        )
                        max_error = max(error, max_error)
                        assert (
                            int(row[model + "_predicted_label"])
                            == expected[model + "_predicted_label"].iat[count]
                        )
                    assert int(row["isFraud"]) == expected.actual_label.iat[count]
                count += 1
        assert count == job["rows_processed"]
        assert max_error < 1e-10, max_error
        metadata = export["metadata"]
        assert (
            not metadata["held_out_membership_verified"]
            and metadata["threshold"] == 0.43
        )
    result = {
        "fixture": path.name,
        "rows": count,
        "upload_mib": path.stat().st_size / 2**20,
        "analysis_seconds": analysis_seconds,
        "export_and_download_seconds": completed - started - analysis_seconds,
        "sampled_peak_working_mib": peak,
        "memory_baseline": baseline,
        "memory_end": memory(pid),
        "maximum_score_error_against_saved_validation": max_error
        if expected is not None
        else None,
        "phases": phases,
        "evaluation": job["evaluation"],
        "zip_mib": zipped.stat().st_size / 2**20,
    }
    client.delete(f"/analyses/{identifier}", headers=headers).raise_for_status()
    path.unlink()
    zipped.unlink()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-pid", type=int, required=True)
    args = parser.parse_args()
    settings = load_settings()
    root = settings.backend_root / "runtime/research_checks"
    root.mkdir(parents=True, exist_ok=True)
    # Preserve selected artifacts and validation reports; never fingerprint test data.
    paths = [
        p
        for folder in (
            settings.research_model_dir,
            settings.reports_dir / settings.research_model_dir.name,
        )
        for p in folder.rglob("*")
        if p.is_file() and p.suffix not in (".lock",)
    ]
    hashes = {str(p): file_sha256(p) for p in paths}
    validation, synthetic, saved = fixtures(settings, root)
    results = []
    with httpx.Client(
        base_url="http://127.0.0.1:8000/api/v1/research", timeout=600
    ) as client:
        for path, expected in ((validation, saved), (synthetic, None)):
            results.append(run(client, path, args.server_pid, expected))
            (root / "capacity.json").write_text(
                json.dumps({"results": results}, indent=2)
            )
    assert all(file_sha256(Path(p)) == h for p, h in hashes.items())
    (root / "capacity.json").write_text(
        json.dumps(
            {
                "results": results,
                "preserved_files": len(hashes),
                "preservation_passed": True,
                "test_used": False,
                "validation_input": "feature-equivalent reconstructed predictors; labels from saved validation report",
            },
            indent=2,
        )
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "rows": [r["rows"] for r in results],
                "test_used": False,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
