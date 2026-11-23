# Experimental three-stage validation

Re-evaluate the original saved candidates using `Experiment-Dataset.csv` as validation data. No training, SMOTE resampling, preprocessing refit, SHAP, official test evaluation, or deployment occurs. All generated files stay in this folder. The runner imports the existing backend package and uses its Python environment; no installation is needed.

The full November 18 run is **complete and independently verified**. It selected **1:100**, **200 trees / depth 20 / minimum leaf size 10**, and a **44%** shared cutoff. See [results review](RESULTS_REVIEW.md), [offline report](outputs/experimental_validation_20261118/report.html), and [run summary](outputs/experimental_validation_20261118/SUMMARY.md). The example run ID below is now occupied; omit it for a fresh run.

From the repository root in PowerShell:

```powershell
& .\backend\.venv\Scripts\python.exe -B .\experimental_three_stage_validation\run.py --run-id experimental_validation_20261118
```

Omit `--run-id` for a unique timestamped name. Existing names are rejected. `--csv <path>` and `--source-run <path>` override the supplied CSV and original `paysim_three_stage_20260928_172539` artifact directory. Relative input paths resolve from the working directory; outputs always go to this folder's `outputs/`. Use a new name after any failed or interrupted run; no automatic resume is provided.

## Sequential protocol

1. Score the eight ratios 1:100, 1:50, 1:20, 1:10, 1:5, 1:3, 1:2, 1:1 at 100 trees/depth 10/minimum leaf 1. Maximize RF-SMOTE AP; ties favor smaller ratios.
2. Freeze that winner and score 12 forests: trees {100, 200}, depth {10, 20}, minimum leaf {1, 10, 50}. Maximize mean RF/RF-SMOTE AP; ties favor shallower depth, fewer trees, then larger leaves.
3. Freeze that pair and maximize mean F1 over the 100 shared cutoffs 1% through 100%. Exact rational ties favor closest to 50%, then the higher percentage. Unrounded scores equal to the threshold predict fraud.

The saved original forest grid exists only at ratio 1:100. If another ratio wins Stage 1, the runner retains its results and reports the exact missing Stage 2 candidates with status `incomplete_missing_models` (exit 3). It never substitutes ratio 1:100 or trains missing candidates. A complete run exits 0. Errors retain `failed` status and a traceback; an interrupted `running` run is incomplete.

## Input and evidence

The CSV must contain exactly the 11 engineered predictors plus binary `isFraud`, with both classes and finite values. Features must already use the saved training representation. Columns are arranged in saved model order, labels never enter prediction, and all rows and duplicates are preserved. `source_row_number` means one-based data-row position, namespaced by the input SHA-256; it is not an original PaySim identifier.

Each output contains an offline `report.html`, `SUMMARY.md`, a manifest, log, verification record, stage candidate tables and decisions, and per-candidate full scores in Parquet. Complete runs also contain the 100-cutoff CSV/Parquet table and SVG plot, final metrics CSV/JSON, and final row-level scores and labels. Precision, recall, F1, MCC, AP, accuracy and confusion counts are recorded; unavailable metrics carry reasons. Stage 1/2 classification metrics use 50% only as supplementary evidence.

The manifest records the validation CSV hash, source/model/code hashes, dependency versions, zero fits, completed stages, elapsed time, and output hashes. Model references are verified before deserialization and source fingerprints are checked again at completion. Saved scores are reused within a run only for an identical model fingerprint on the same CSV. The independent verifier recomputes candidate metrics, selections and all threshold confusion counts from saved predictions.

Results use the same experimental validation data for selection and reporting. Dataset independence and overlap have not been established; these are not independent test results. No application settings or prior artifacts change.

## Verification

```powershell
& .\backend\.venv\Scripts\python.exe -B -m pytest -c .\experimental_three_stage_validation\pytest.ini .\experimental_three_stage_validation\tests --basetemp .\experimental_three_stage_validation\runtime\pytest -q
& .\backend\.venv\Scripts\python.exe -B .\experimental_three_stage_validation\verify.py .\experimental_three_stage_validation\outputs\experimental_validation_20261118
```

Tests cover CSV errors and preserved identities, label exclusion, deterministic tie rules, direct cutoff verification, complete and missing-model paths with fitting/resampling forbidden, non-overwriting runs, corruption rejection, and retained failure reports.

Optional offline browser verification, using the existing frontend Playwright installation and installed Chrome:

```powershell
node .\experimental_three_stage_validation\tests\preview_report.cjs .\experimental_three_stage_validation\outputs\experimental_validation_20261118
```

Screenshots and browser verification evidence go to `runtime/qa/` inside this folder.
