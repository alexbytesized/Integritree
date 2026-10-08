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

A backend restart invalidates browser session tokens. If a new CSV upload receives
HTTP 410, the browser clears the expired session and analysis reference, creates a
new session, and retries the selected file once automatically. A second 410 remains
a visible, manually retryable error. Validation, size, queue, and connection errors
are not automatically retried, avoiding duplicate analyses after uncertain network
failures. Previously completed temporary analyses cannot survive a backend restart.

The trusted model path defaults to `artifacts/paysim_three_stage_20260928_172539`;
the training-background preparation defaults to `data/prepared/paysim_phase2_20260918`.
Override them with `INTEGRITREE_RESEARCH_MODEL_DIR` and
`INTEGRITREE_RESEARCH_PREPARED_DIR`. Frozen selection verification happens once,
on first analysis. The application applies the saved common threshold (43% for
the current selection), using `score >= threshold`, without fitting anything.

## CSV inputs and capacity

Use UTF-8 CSV with unique, case-sensitive headers. Column names automatically
select the raw or prepared format; column order may vary.

Raw uploads require these six columns:
`step,type,amount,nameOrig,nameDest,isFraud`. The download-template link provides
demonstration data, not test data. The other five original PaySim columns are
optional source information; arbitrary additional columns are rejected.

- `step`: positive integer in the saved predictor validator's supported range.
- `type`: CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER, or unknown. Blank means unknown.
- `amount`: finite nonnegative number, or blank for saved training-median imputation.
- `nameOrig`, `nameDest`: C/M-prefixed entity identities, as required by the saved validator.
- `isFraud`: complete binary 0/1 labels; never a predictor. No label imputation.

Prepared uploads require exactly these twelve columns, with no index or extra
columns:

```text
hour_of_day,day_of_week,type_CASH_IN,type_CASH_OUT,type_DEBIT,type_PAYMENT,type_TRANSFER,log_amount,is_zero_amount,is_merchant_origin,is_merchant_dest,isFraud
```

These eleven features must already use the selected model bundle's preprocessing,
including scaling. Prepared uploads bypass preprocessing entirely. Header names
identify the format but do not prove how the file was processed. All predictors
must be finite numbers without missing values. Binary indicators must be 0 or 1;
at most one transaction-type indicator may be 1, with all-zero meaning unknown.
Scaled numeric values may fall outside `[0, 1]`; the uploader does not clip them.
`isFraud` remains required ground truth and never enters the models.

To create a CSV from an internal prepared split, use `load_prepared_split` to obtain
aligned features and labels, add the labels as `isFraud`, and export with
`index=False`. Do not include `source_row_number`, and preserve feature/label row
alignment. Mixed raw/prepared layouts are rejected.

Prepared uploads show `N/A (file uploaded is already preprocessed)` for unavailable
Original Inputs, while retaining the ground-truth label. Derived Inputs reverses
scaling for readable display only. Model predictions and SHAP use the supplied
prepared features. Neither format verifies that uploaded rows belong to the
official held-out test set.

Status, record detail, and export metadata identify `input_format` as `raw` or
`prepared`. Export metadata records `preprocessing_applied`. The existing ZIP and
CSV filenames are retained; the CSV preserves the actual uploaded columns and
values alongside predictions for both formats.

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
`search_field=row_number` selects the interface behavior. The contribution chart endpoint (legacy `/waterfall/` route)
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

Download results creates an asynchronous `integritree_results.zip` for the complete
upload, containing exactly two files:

- `Experiment-Paper_YYYY-MM-DD.pdf`: the supplied experiment paper with its five
  tables filled from the same full-upload evaluation. All explanatory notes are
  placed under the template's final **Notes:** heading.
- `Raw-Results_YYYY-MM-DD.csv`: stable upload-hash/row identities, source fields, paired
  full-precision scores and predicted labels. Explanation status and narratives
  are excluded, including when SHAP has already been requested. All rows remain
  in upload order with the existing CSV safety handling.

Both filenames use the same Philippine date (UTC+8), captured once before PDF
conversion starts, even when generation crosses midnight. A later export uses
its own start date. JSON files and the HTML report are no longer included or
rendered for download. Evaluation and explanation data remain available to the
application, and the export-status response retains its metadata, including
`scope: uploaded_dataset` and `held_out_membership_verified: false`.
Previously downloaded ZIPs are unchanged.
The researcher establishes final-test provenance in the research record. Scores
are model outputs; display risk bands do not establish real-world calibration.
CSV text that could execute as a spreadsheet formula is prefixed with an apostrophe.
Filtered exports and XLSX are not implemented. The paper is exported as PDF;
the intermediate Word document is temporary and is not included in the ZIP.

### Experiment paper PDF setup

The retained template is
`backend/src/integritree/templates/Experiment-Paper-Template.docx` and is included
in the Python package. Keep its five result tables and their labels. Answer cells
may be blank; Tables 1-2 also support `TP`, `FN`, `FP`, `TN` in their corresponding
positions, and Table 4 supports `A`, `B`, `C`, `D` in row order. Other prefilled
values are rejected. Keep the final `Notes:` heading followed by an empty paragraph. The exporter
changes only answer cells and the Notes area in a temporary copy. It preserves
the supplied wording, equations, headers, styles, and page settings. Replacing the
template with a different structure requires updating the table mapping. Generated
answer numbers remain italic and non-bold. Each export reads the current template;
after replacing it, use Download Results again to generate an updated PDF.

From the project root on Windows, run:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File backend/scripts/setup_research_pdf.ps1
```

This downloads the pinned LibreOffice installer, verifies its SHA-256 and signed
publisher, and extracts a dedicated runtime under `backend/runtime/document_tools`.
It does not install LibreOffice system-wide or change the persistent PowerShell
execution policy. Windows Installer must be able to use its temporary directories.
The converter default is `runtime/document_tools/libreoffice/program/soffice.com`,
relative to the backend root. An alternate LibreOffice executable can be selected
with `INTEGRITREE_RESEARCH_PDF_CONVERTER` in the environment or backend `.env`.
Restart the backend after changing settings. No Word installation is required.

Conversion runs in the existing export worker with a separate temporary LibreOffice
profile and a 120-second timeout. A missing converter, incompatible template, or
failed conversion fails the export with an actionable message; it never produces
a successful ZIP without the paper. Correct the problem and retry Download Results.
Temporary document/profile files are removed after each conversion, including
failure. They use short temporary `session_*` directories directly under the
research session root to avoid Windows path-length limits; startup removes any
left by an interrupted process. Normal session cleanup covers other generated files.

Counts are integers; percentage metrics and percentage comparisons use two decimal
places, MCC and chi-squared use four. Small nonzero p-values use scientific notation
when four decimals would round them to zero. `N/A` cells are explained only in
Notes, as are coefficient differences for negative MCC and McNemar edge cases.
The primary McNemar result remains in Table 5; any applicable exact-binomial
supplement is in Notes. This formatting does not change the full-precision JSON.

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
real SHAP contribution bar charts, return navigation, ZIP download, clear results, malformed
input/retry, and expired sessions. Browser diagnostics go to ignored `test-results`.

Final verification passed **208 backend tests** (five upstream warnings), all
**three browser checks**, frontend lint/build, dependency checks, document links,
and Git whitespace checks. That milestone used the earlier four-file ZIP contract;
the current download contract is the two-file PDF/CSV archive documented above.
The earlier ZIP was inspected for
43% threshold, unverified membership, full record count, and actual SHAP coverage.
Receipt workflow readiness and final test evaluation are outside this milestone.
