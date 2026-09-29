# Researcher workflow

The researcher view uses one labeled CSV analysis flow. It does not identify or
verify the reserved test set. The researcher is responsible for withholding that
set until receipt integration, whole-tool verification, and explicit authorization
for the final experiment are complete. Perform that final evaluation through the
tool and download its results. Do not tune settings against those results.

## Running locally

From `backend`, run:

```powershell
& ./.venv/Scripts/python.exe -m uvicorn integritree.api.main:create_app --factory --host 127.0.0.1 --port 8000
```

From `frontend`, run `npm.cmd run dev -- --host 127.0.0.1`, then open
`http://127.0.0.1:5173/researcher-upload`. Vite proxies `/api` to port 8000.
Use one backend process; multiple workers cannot share temporary analyses. An OS
lock prevents a second process from deleting the first process's session files.

The trusted model path defaults to `artifacts/paysim_three_stage_20260928_172539`;
the training-background preparation defaults to `data/prepared/paysim_phase2_20260918`.
Override them with `INTEGRITREE_RESEARCH_MODEL_DIR` and
`INTEGRITREE_RESEARCH_PREPARED_DIR`. Frozen selection verification happens once,
on first analysis. The application applies the saved common threshold (43% for
the current selection), using `score >= threshold`, without fitting anything.

## CSV inputs and capacity

Use UTF-8 CSV with unique headers and these six required columns:
`step,type,amount,nameOrig,nameDest,isFraud`. The download-template link provides
demonstration data, not test data. The other five original PaySim columns are
optional source information; arbitrary additional columns are rejected.

- `step`: positive integer in the saved predictor validator's supported range.
- `type`: CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER, or unknown. Blank means unknown.
- `amount`: finite nonnegative number, or blank for saved training-median imputation.
- `nameOrig`, `nameDest`: C/M-prefixed entity identities, as required by the saved validator.
- `isFraud`: complete binary 0/1 labels; never a predictor. No label imputation.

Invalid input rejects the entire analysis. No partial metrics or records are
published. Errors report columns and one-based data-record numbers, excluding the
header; the first failing batch includes at most 20 example positions per column.
Quoted multiline records count as one record. Duplicate and missing headers,
inconsistent field counts, invalid UTF-8, and malformed quoting are rejected.

The limit is **500 MiB**, enforced on actual streamed bytes even without a
Content-Length header. There is **no fixed record-count cap**. Processing batches
contain at most 25,000 records and are also flushed at 8 MiB of field characters.
The CSV parser rejects fields above its 131,072-character field limit. One analysis runs
at a time; one additional upload may wait. A full queue returns HTTP 429.

SQLite keeps records, searches, pagination, exact tied-score Average Precision,
and exports on disk. Full-upload metrics are independent of table filters.
SHAP is requested for displayed records and opened details, including legitimate
predictions; it is not automatically computed for an entire large upload.

The interface displays Transaction IDs as one-based row numbers scoped to the
current upload. Its search matches those row numbers. Internal and exported IDs
remain `<upload SHA-256>:<row number>`. The records API defaults to full-ID search;
`search_field=row_number` selects the interface behavior. The waterfall endpoint
accepts `presentation=row_number` for a separately cached display chart without
changing the original chart or recomputing SHAP. Full charts open only through
the model's “See Full SHAP Evaluation” modal.

Weekday names use Monday=0 through Sunday=6 as a display convention for the
simulation cycle, not verified calendar dates. Display scores use two decimal
places; classification and risk-band selection use unrounded scores.

## Temporary storage and downloads

Each browser tab receives an opaque session token stored in sessionStorage and
sent in `X-Research-Session`. Refresh preserves it. Another browser session cannot
read, explain, download, or delete its analyses. This is a local research tool,
not an internet-facing authenticated service.

Session files live only under `backend/runtime/research_sessions/session_*`.
Stopping the backend cleans current data; startup cleans interrupted sessions.
Existing model, prepared-data, and validation-report directories are excluded.
Clear results deletes the analysis after its active work finishes. Closing the
browser alone does not stop the backend; its temporary files persist until backend
shutdown/startup cleanup. Reopening a closed tab may require uploading again.

All new analyses, including the eventual official evaluation, use this retention
policy. **Download results before stopping/restarting the backend.** There is no
automatic permanent archive of uploaded analyses.

Download results creates an asynchronous ZIP for the complete upload:

- `results.csv`: stable upload-hash/row identities, source fields, paired full-precision
  scores and labels, explanation status, and computed narratives.
- `evaluation.json`: confusion matrices, precision, recall, F1, MCC, Average
  Precision, accuracy, descriptive comparisons, and paired McNemar results.
- `metadata.json`: upload fingerprint, model run and fingerprint, settings,
  threshold, timestamp, evaluated population, and actual SHAP coverage.
- `report.html`: escaped, offline evaluation and provenance report.

Exports state `scope: uploaded_dataset` and `held_out_membership_verified: false`.
The researcher establishes final-test provenance in the research record. Scores
are model outputs; display risk bands do not establish real-world calibration.
CSV text that could execute as a spreadsheet formula is prefixed with an apostrophe.
Filtered exports and PDF/XLSX are not implemented in this milestone.

## Capacity evidence

Measured on this Windows workstation through the HTTP upload/export routes:

| Workload | CSV size | Analysis | Export/download | Memory |
| --- | ---: | ---: | ---: | --- |
| 636,262 validation-derived records | 35.46 MiB | 52.53 s | 9.67 s | 880.23 MiB process peak, including first bundle verification |
| 1,000,000 synthetic records | 44.48 MiB | 41.72 s | 15.72 s | 298.93 MiB sampled working-set peak with cached bundle |

The validation CSV reconstructs equivalent predictors from validation features;
its labels come only from the saved validation report. Every exported score and
prediction matched the saved validation output exactly (maximum score error 0).
389 selected-run and validation-report files retained their hashes. Held-out test
records were not read or evaluated. Full-population SHAP is not included in these
timings. These measurements do not establish an unlimited processing capacity or
guarantee performance for other machines, record widths, or files near 500 MiB.

Reproduce with `scripts/benchmark_research.py --server-pid <backend-process-id>`
from `backend`. It creates disposable validation-derived and synthetic inputs,
checks downloaded rows, cleans its analyses/fixtures, and writes capacity evidence
under `runtime/research_checks/capacity.json`.

## Automated checks

Run backend tests with `.venv/Scripts/python.exe -m pytest` from `backend`.
From `frontend`, run `npm.cmd run lint` and `npm.cmd run build`. With both local
servers running, `npm.cmd run test:e2e` runs Playwright using installed Microsoft
Edge and isolated browser contexts. It uses synthetic CSVs with the real frozen
models: template/upload, refresh, pagination/filter/search, original record details,
real SHAP waterfalls, return navigation, ZIP download, clear results, malformed
input/retry, and expired sessions. Browser diagnostics go to ignored `test-results`.

Final verification passed **208 backend tests** (five upstream warnings), all
**three browser checks**, frontend lint/build, dependency checks, document links,
and Git whitespace checks. The real browser ZIP was inspected for all four files,
43% threshold, unverified membership, full record count, and actual SHAP coverage.
Receipt workflow readiness and final test evaluation are outside this milestone.
