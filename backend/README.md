# Integritree backend

## Validation workflow

The sole selection workflow is [three-stage validation](../docs/THREE_STAGE_VALIDATION.md):
SMOTE ratio by RF-SMOTE validation Average Precision, shared forest settings by
mean validation AP, then a common threshold by exact mean validation F1. Each
stage freezes before the next. Both models are retained; selection does not pick
a model winner. Stage 1 froze 1:100; Stage 2 selected 100 trees/depth 10/leaf 1.
Stage 3 selected the shared 43% cutoff from exactly 1%, 2%, ..., 100%. See
[run status](../docs/VALIDATION_RUN_STATUS.md) for measured outcomes.

Preparation, paired RF/RF-SMOTE training, saved inference, evaluation, three-stage
selection, SHAP, and the researcher HTTP/frontend workflow are implemented.
See [researcher operation and retention](../docs/RESEARCHER_WORKFLOW.md).
The local receipt audit/OCR comparison and internal confirmed-input mapping/paired
prediction are implemented. See [receipt mapping contracts](../docs/RECEIPT_MAPPING.md).
The [receipt API/frontend connection](../docs/RECEIPT_APPLICATION.md) now includes local
OCR, confirmation, paired predictions/SHAP, downloads and temporary cleanup. Final test evaluation waits
until the complete tool is ready and the researcher explicitly authorizes it.

The approved receipt scope now targets selected GCash app workflows across all five PaySim categories,
with local OCR and temporary application storage. Save development screenshots in
`data/raw/receipt_samples/{TRANSFER,CASH_IN,CASH_OUT,PAYMENT,DEBIT,UNSURE}/`;
this ignored collection is retained separately from temporary application uploads.
See [receipt decisions and collection instructions](../docs/RECEIPT_WORKFLOW.md).
PAYMENT includes wallet-funded QR and Pay Online. Personal-wallet role mappings
for TRANSFER/PAYMENT/DEBIT are settled; cash-agent mappings and exact layouts remain
pending. RapidOCR is provisionally selected from the
[local benchmark](../docs/RECEIPT_OCR_BENCHMARK.md), which records remaining date/role
failures and reproducible isolated setup. One-image 10 MiB / 20-million-pixel
PNG/JPEG uploads and individual ZIPs are implemented for Express Send, Pay Online
and confirmed bank-account transfers. Cash categories and merchant QR remain pending.

## Install on Windows

Use Python 3.12. The development environment was verified with Python 3.12.5
on Windows. Run these PowerShell commands from the `backend` directory:

```powershell
py -3.12 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --require-hashes -r requirements-dev.lock
& .\.venv\Scripts\python.exe -m pip install --no-deps --no-build-isolation -e .
```

If the Python launcher is unavailable and Python 3.12 was installed in its
standard per-user location, create the environment with:

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" -m venv .venv
```

The lock pins development/runtime dependencies and hashes. It includes the build
tools needed by the editable installation. Keep this project environment separate
from Anaconda or other projects. Environment activation is optional because the
commands explicitly select its interpreter.

To update dependencies intentionally, edit `pyproject.toml`, regenerate the lock,
install it, and rerun the tests:

```powershell
& .\.venv\Scripts\python.exe -m piptools compile --extra dev --generate-hashes --allow-unsafe --strip-extras --no-emit-index-url --constraint configs/model-dependency-constraints.txt --output-file requirements-dev.lock pyproject.toml
```

The lock was generated on Windows with Python 3.12; installation on other
platforms has not been verified. The constraints preserve saved-model dependency
versions for preparation, training, evaluation, and explanations. Ruff is a pinned
development-only formatter and does not change the saved-model runtime.
OCR benchmark dependencies are isolated in
`runtime/receipt_tools/venv`, pinned in `requirements-ocr-benchmark.lock`; do not
install that snapshot into the ML environment.

## Code formatting

Run these commands from `backend/`. The [Ruff formatter](https://docs.astral.sh/ruff/formatter/)
uses four spaces, double quotes, an 88-column target, and Python 3.12 syntax.
Its configuration applies only to the backend. Generated data, models, reports,
and runtime evidence are excluded; the frontend has no formatting changes.

```powershell
# Format maintained backend Python files.
& ./.venv/Scripts/python.exe -m ruff format src scripts tests
# Check formatting without editing files.
& ./.venv/Scripts/python.exe -m ruff format --check src scripts tests
```

Comments explain invariants and non-obvious decisions rather than development
history. Preserve recorded artifact hashes and dependency versions when cleaning
source; historical fingerprints describe the code used for the recorded run.
The OCR benchmark checks exact source hashes when reusing a development freeze.
After source changes, create a new development benchmark before verification;
keep earlier benchmark records and their fingerprints unchanged.

The one-time Stage 3 reset utility and unused prediction scaffolds are retired.
Use the researcher CSV or receipt APIs for application inference. Selection still
supports pause/resume with frozen evidence, and incomplete historical resets are
rejected. Historical reset records remain in the validation evidence.

## Test and run

```powershell
& .\.venv\Scripts\python.exe -m pytest
& .\.venv\Scripts\python.exe -m pip check
& .\.venv\Scripts\python.exe -m uvicorn integritree.api.main:create_app --factory --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000/api/v1/health for the health response or
http://127.0.0.1:8000/docs for the generated API documentation. Stop the server
with Ctrl+C. From `frontend`, run `npm.cmd run dev -- --host 127.0.0.1` and open
`http://127.0.0.1:5173/researcher-upload`. Vite proxies both APIs to port 8000. Open `/upload` for the receipt workflow.
Receipt extraction needs the isolated OCR environment described in the
[application setup guide](../docs/RECEIPT_APPLICATION.md). Restart an already-running
backend to load newly added routes.

Tests use synthetic records and temporary configuration files. Neither tests
nor health checks require loading the real dataset or a saved model. The health
endpoint indicates that the application is available, not that inference is ready.

## Local settings

Defaults work for an editable installation in this checkout. Optionally copy
`.env.example` to `.env` and adjust paths or the explicit frontend CORS origins.
Do not overwrite an existing local configuration.

`load_settings()` reads the backend root's `.env`. Process environment values
override that file. Relative paths resolve against the backend root regardless
of the shell's working directory. Settings validation does not create directories.

To use another backend working directory, set the absolute
`INTEGRITREE_BACKEND_ROOT` in the process environment before starting the command.
Do this for a non-editable installation too; configuration/data are external to
the installed package. Setting the root inside `.env` does not relocate that file.

## Experiment configuration

`configs/experiment.yaml` records the dataset identity and confirmed methodology.
Unresolved research choices are explicitly `null`.

The approved signed comparison is explicit in this active file. Evaluation and
explainer policies are saved separately against immutable model-run provenance;
do not edit an old bundle's configuration to enable a later phase. Ordinary model bundles preserve fixed-parameter checks; completed three-stage
selections have a separate validated schema-2 contract.

```powershell
& .\.venv\Scripts\python.exe -m integritree.config
& .\.venv\Scripts\python.exe -m integritree.config --require-stage prepare
```

The first command validates the draft and lists unresolved fields (exit 0).
The second now passes (exit 0): the researchers approved the Phase 2 preparation
settings. Training, evaluation, and explanation readiness also pass with the
approved active configuration. Available stages are `prepare`, `train`,
`evaluate`, and `explain`; requirements include earlier stages.
`--config` accepts an absolute path or one relative to the backend root.

Validation does not approve a methodology or execute research. Preparation and
baseline training, evaluation, selection, and explanation commands execute their
documented workflows. Researcher batch analysis and exports are available through
the application API. No command produces mock research results or modifies the raw
dataset.

## Prepare the approved PaySim dataset

From `backend`, after installation:

```powershell
& .\.venv\Scripts\python.exe scripts/prepare_data.py
```

The equivalent module command is `python -m integritree.ml.preparation`.
`--config` selects another configuration (relative to the backend root), and
`--run-id` selects an unused output name. The dataset path comes from local
settings. Default output is `data/prepared/prepare_<timestamp>_<suffix>/`.
Existing runs are never overwritten. Progress goes to stderr; final success JSON
goes to stdout. Failure exits 2 and leaves a failed audit/metadata record when
an output directory has already been created. No trained model is required.

The command verifies the approved dataset hash/row count, validates all source
columns, removes exact duplicates, saves stratified split membership, fits amount
imputation and Min-Max scaling on original training records only, and applies
those fitted values to each split. It does not apply SMOTE or train RF models.

CSV validation/feature output use bounded batches. A temporary SQLite database
compares complete canonical typed source records across batches, so duplicate
removal does not rely on probabilistic row hashes. Blank fields are missing;
literal invalid text/nonfinite numbers are errors. Excluded numeric balances and
flags remain in retained source data, with missingness audited; they are never
imputed for or included in the feature matrix. Original row IDs survive cleaning.

Splitting uses an 80/20 stratified call followed by an equal stratified split of
the held-out 20%, both with seed 42. Ratios follow integer rounding and class-count
constraints. Both classes must occur in every split. Neither duplicate removal
nor stratification operates on the engineered feature vectors.

| Bundle file                   | Contents                                                                                                              |
| ----------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `metadata.json`               | Completion status, dataset/run identity, split counts, package versions, code/file hashes.                            |
| `configuration.json`          | Exact validated experiment settings used for preparation.                                                             |
| `audit.json`                  | Source hash, scanned/retained counts, missingness, numeric ranges, class/type counts, duplicate count, failures.      |
| `source.parquet`              | Retained typed source columns plus original `source_row_number`; includes categorical data for later SMOTE decisions. |
| `duplicates.parquet`          | Removed original row numbers and the first matching row number.                                                       |
| `split_manifest.parquet`      | Original row number, split name, and separate `actual_label`.                                                         |
| `preprocessing.json`          | Training amount median, scaler coefficients/extrema, feature order, and training count.                               |
| `train_features.parquet`      | Training model features plus row identity; no labels/balances/raw source features.                                    |
| `validation_features.parquet` | Validation features with the same schema and transformations.                                                         |
| `test_features.parquet`       | Test features with the same schema and transformations.                                                               |

`source_row_number` is an identifier, never a predictor. Combine it with the
bundle's dataset SHA-256 to recover the full transaction ID. Use
`load_prepared_split(bundle, "train")` from `integritree.ml.preparation` to obtain
aligned `X`, `y`, and source IDs with fingerprint checks. That reader materializes
one split in memory; preparation itself is batched. Consumers must require
`metadata.status == "complete"`; failed/partial runs are not usable artifacts.

`FittedPreprocessor.load(path).transform(inputs)` accepts exactly the five source
attributes and reuses the saved training state. It accepts missing amount/type
under the approved policy; missing timing/entity inputs and extra target/balance
columns fail validation. Shared inference loads the fitted model bundle and reuses these transformations.

Memory use is bounded for parsing, deduplication, and feature writing. Splitting
still allocates arrays proportional to the number of records; training-median
calculation uses a disk-backed scratch array. Allow several GB of free disk for
scratch data and outputs. A repeated preparation creates another bundle, not a
replacement. Run IDs/timestamps differ; split membership and fitted state are
reproducible for identical source/configuration/dependency versions.

## Verified local preparation

The supplied PaySim file was prepared as `paysim_phase2_20260918`. All 6,362,620
records were retained: no exact duplicates, missing fields, or validation errors.
Training contains 5,090,096 records (6,570 fraud), validation 636,262 (822 fraud),
and testing 636,262 (821 fraud). No model training or SMOTE was performed.

All 106 tests passed. Separate full-output verification checked file/source hashes,
all row identities, split/label alignment, finite feature values, binary indicators,
and saved-preprocessing replay on 100 records per split. See
`reports/paysim_phase2_20260918/verification.json` for the local verification record.
The raw CSV is unchanged. Generated artifacts are ignored by Git.


## Training, selection, and saved inference

`configs/experiment.yaml` supplies data, features, seeds, and fixed training
parameters. Its standalone trainer defaults are not selected research settings.
The three-stage protocol overrides ratio and forest candidates; 0.50 is only a
supplementary classification checkpoint until Stage 3 freezes a threshold.

The current run completed all three stages and selected a 43% cutoff; it does
not need another selection run. To continue an interrupted run using its saved
models and validation evidence, replace `<run-id>` below with that run's ID:

```powershell
& ./.venv/Scripts/python.exe scripts/select_three_stage.py --prepared data/prepared/paysim_phase2_20260918 --config configs/experiment.yaml --protocol configs/validation_three_stage.yaml --resume "artifacts/<run-id>" --jobs 1 --stop-after-stage 3
```

Installed commands `integritree-select` and `integritree-select-three-stage` invoke
the same selector. `--stop-after-stage` accepts 1 (ratio), 2 (forest), or 3
(threshold); default 3 runs the full workflow.
Resume the same frozen run after failure or graceful pausing; never run two workers.
Memory checks remain enabled and the data/settings must not be changed to pass them.

For an active selection worker, request a graceful pause from a separate terminal
using its run ID:

```powershell
& ./.venv/Scripts/python.exe scripts/pause_selection.py --run "artifacts/<run-id>"
```

The command reports `pause_requested`. Wait for worker metadata `status: paused`
and worker exit before closing the training session or shutting down. Both model
fits and validation for the active candidate finish before its checkpoint is saved
and the worker exits. A final candidate instead completes and freezes the stage.
Repeated requests are safe; requests to inactive/completed runs are rejected.
Resume using the intended stop stage; completed candidates are verified and reused.
Abrupt interruption does not preserve partial model fits and can require retraining
an unfinished candidate. Changing when candidates run does not change their training
inputs or selection rules. Mid-fit checkpoints are not implemented.

A completed Stage 2 produces `status: awaiting_next_stage`, `completed_stage: 2`,
and `stage: forest_selected`. The `reports/<run-id>/stage2_review/` export includes
all 12 candidates, both AP values, mean AP, the frozen decision, and provenance.
No threshold or final selected pair exists until Stage 3. A paused run exports no
completed-stage report for its unfinished stage. See the
[complete protocol](../docs/THREE_STAGE_VALIDATION.md) and
[current run status](../docs/VALIDATION_RUN_STATUS.md).

The standalone `scripts/train_models.py` command remains available for explicit
fixed-parameter fits. It is not a selection workflow. Ordinary bundles contain
both joblib models, preprocessing, configuration, preparation provenance, training
audit, reload verification, and fingerprinted metadata. Load only trusted local
joblib bundles. Incomplete bundles and mismatched hashes/dependencies are rejected.

`load_bundle` accepts ordinary fitted pairs and completed schema-2 three-stage
selections. Old two-step selections are rejected. Shared inference applies saved
preprocessing and returns both continuous scores and thresholded labels.

## Evaluation and explanations

`scripts/evaluate_models.py` writes paired predictions, metrics, comparisons,
McNemar results, figures, and provenance. Official test evaluation requires a
completed three-stage selection. A Stage 1 run is not a final selected bundle.
No test evaluation is included in the current execution.

`scripts/explain_models.py` explains an existing report using the same saved
models and a shared original-training background. Coverage is explicitly preview,
sampled global, or full report. SHAP is not triggered by validation selection.
See [method and integrity details](../docs/PHASE4_IMPLEMENTATION.md).

## Generated files and references

Raw/prepared data, models, reports, runtime outputs, environments, and secrets are
local and excluded from Git. `data/prepared/` retains preprocessing and split
provenance; `artifacts/` holds fitted pairs and selection checkpoints; `reports/`
holds exported evidence; `runtime/validation/` holds current worker logs.

- [Methodology](../docs/METHODOLOGY.md)
- [Thesis context](../docs/THESIS_CONTEXT.md)
- [Application implementation plan](../docs/BACKEND_IMPLEMENTATION_PLAN.md)
- [API contracts](../docs/API.md)
- [Manuscript alignment checklist](../docs/THESIS_DOCUMENT_CHANGES.md)
