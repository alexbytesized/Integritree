# Integritree backend

## Current validation protocol: three stages

The revised workflow selects SMOTE ratio by RF-SMOTE validation AP, one of 12
shared forests (including minimum leaf size) by mean validation AP, then a common
cutoff by exact mean validation F1 over all distinct scores. It retains both
models and never evaluates test data. Historical selectors/artifacts remain intact.
See [protocol, outputs, and resume instructions](../docs/THREE_STAGE_VALIDATION.md).

Current execution status: all 172 backend tests passed. Original Stage 1 completed
and froze 1:10. The separately authorized [revision-2 extension](../docs/STAGE1_RATIO_EXTENSION_V2.md)
adds 1:20, 1:50, and 1:100 and selects across all eight ratios. It started at
00:53 Manila time on 2026-09-28; the original five candidates were reused and
1:20 began training. The new job stops before Stage 2 or threshold selection.
Check [run status and logs](../docs/VALIDATION_RUN_STATUS.md) for current evidence.

The revision-2 job has already started. If interrupted, resume it from `backend/`:

```powershell
& ./.venv/Scripts/python.exe scripts/select_three_stage.py --prepared data/prepared/paysim_phase2_20260918 --config configs/experiment.yaml --protocol configs/validation_three_stage_v2.yaml --resume artifacts/paysim_stage1_ratio_extension_v2_20260928 --jobs 1 --stop-after-stage 1
```

The sections below document historical workflows; their test commands require a
separate authorization and are not part of the current validation-only execution.

Phases 1-4 code is implemented: configuration, a health endpoint, PaySim
preparation, matched RF/RF-SMOTE baseline training, saved artifacts, and shared
individual/batch inference. Full PaySim preparation and baseline training are
complete and verified. Both models are saved in artifacts/paysim_phase3_baseline_20260920/.
Phase 4 adds evaluation, validation selection, and SHAP. Full baseline validation,
the four-candidate 1:1 validation search, and the validation-only 1:3 sensitivity
study have run on 636,262 records; five validation records have verified explanations
for both baseline models. The 1,000-record global SHAP job and final test evaluation
have not run. Prediction APIs and OCR remain future work.

The 2026-09-22 [alignment audit](../docs/BACKEND_ALIGNMENT_AUDIT.md) checked this
foundation against the completed mock-up review: 125 tests passed. Active evaluation
configuration now selects signed_over_mean; legacy saved configurations remain
readable and unchanged. Phase 4 now implements the metric calculations.
The next software phase is application services/API; expanded SHAP details require only a
waterfall graph. The numerical table is a suggestion, not an implementation target.

Read [the thesis context](../docs/THESIS_CONTEXT.md),
[the implementation plan](../docs/BACKEND_IMPLEMENTATION_PLAN.md),
[methodology decisions](../docs/METHODOLOGY.md), and
[API contracts](../docs/API.md) before developing the next phase.

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
platforms has not been verified. Phase 2 adds NumPy, pandas, PyArrow, and scikit-learn. Phase 3 adds imbalanced-learn 0.14.2 and explicit joblib support.
Phase 4 adds SHAP 0.52.0, Matplotlib, and explicit SciPy support. The constraints
preserve saved-model dependency versions. OCR libraries remain future work.

## Test and run

```powershell
& .\.venv\Scripts\python.exe -m pytest
& .\.venv\Scripts\python.exe -m uvicorn integritree.api.main:create_app --factory --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000/api/v1/health for the health response or
http://127.0.0.1:8000/docs for the generated API documentation. Stop the server
with Ctrl+C. The frontend is not connected to this backend yet.

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
do not edit an old bundle's configuration to enable a later phase. Phase 3 bundle
loading preserves baseline checks; Phase 4 adds a separate validated selected-run contract.

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
documented workflows. The general batch-export script remains a guarded
placeholder (exit 2). No command produces mock
research results or modifies the raw dataset.

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
columns fail validation. No inference model is implemented yet.

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

## Directory responsibilities

| Location                       | Responsibility                                                                           |
| ------------------------------ | ---------------------------------------------------------------------------------------- |
| `configs/`                     | Experiment configuration and seeds.                                                      |
| `scripts/`                     | Preparation and baseline training commands; evaluation/batch-export placeholders.        |
| `src/integritree/settings.py`  | Local paths and environment settings.                                                    |
| `src/integritree/config.py`    | Experiment schema, validation, and stage requirements.                                   |
| `src/integritree/contracts.py` | Shared transaction/result/provenance contracts.                                          |
| `src/integritree/api/`         | HTTP interfaces; currently health only.                                                  |
| `src/integritree/services/`    | Future workflow coordination.                                                            |
| `src/integritree/ml/`          | Data, preprocessing, training, artifacts, and paired inference; evaluation/SHAP pending. |
| `src/integritree/receipts/`    | Future GCash extraction and confirmed-input mapping.                                     |
| `tests/`                       | Synthetic foundation, preparation, training, artifact, and inference tests.              |
| `data/raw/`                    | Original local PaySim CSV; preserve the source.                                          |
| `data/prepared/<run_id>/`      | Local preparation bundles, audits, and split membership.                                 |
| `artifacts/<run_id>/`          | Models, preprocessing, configuration, audit, and provenance bundles.                     |
| `reports/<run_id>/`            | Future predictions, metrics, statistics, and figures.                                    |
| `runtime/`                     | Future temporary uploads and exports.                                                    |

Scripts and application services will share the `ml/` implementation.
Application inference must reuse saved models and fitted transformations.

Git ignores datasets, environments, model binaries, generated reports, uploads,
and secrets. Commit source, reviewed configuration, the dependency lock, and
small synthetic fixtures. Do not commit personal receipts.

## Foundation references

- [Python project configuration](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/)
- [Pydantic settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)
- [Dependency locking with pip-tools](https://pip-tools.readthedocs.io/en/latest/)
- [FastAPI testing](https://fastapi.tiangolo.com/tutorial/testing/)

## Phase 3: baseline training and saved inference

Run from the backend directory:

```powershell
& .\.venv\Scripts\python.exe -m integritree.config --require-stage train
& .\.venv\Scripts\python.exe scripts/train_models.py --prepared data/prepared/paysim_phase2_20260918 --run-id paysim_phase3_baseline_20260920 --jobs 1
```

The installed entry point integritree-train accepts the same arguments.
Paths are relative to the backend root unless absolute. Omit --run-id for a
generated unique name. Existing run directories are never overwritten.

Current config selects the approved 100-tree/depth-20 matched baseline, ordinary
SMOTE at 1:1 with k=5, model/SMOTE seeds 42, and common >=0.50 fraud cutoff.
Bootstrap is enabled and class weighting disabled. Only original training records
are loaded. No validation tuning or test evaluation is performed.

The full run completed successfully on 2026-09-20 after unused applications were
closed. The retry passed the memory check with 4.94 GiB available; the initial
attempt had stopped at 0.32 GiB. Both models used the approved settings/data.
Recorded training duration: 9,594.625 seconds (about 2 h 40 min).
For future runs, the estimated working arrays need 4.65 GiB plus tree/runtime
headroom; this is not a guaranteed peak-memory bound. The automatic physical-memory
check is Windows-specific. Training materializes the original matrix and SMOTE
output without silently reducing data or changing settings.

The run ID shown above now exists. For a new run, omit --run-id or choose a new
name; existing bundles are never overwritten.

A successful run writes these files under artifacts/<run_id>/:

| File                       | Meaning                                                                       |
| -------------------------- | ----------------------------------------------------------------------------- |
| rf.joblib, rf_smote.joblib | Both fitted baseline classifiers.                                             |
| preprocessing.json         | Original training-fitted transformations; no refit.                           |
| configuration.json         | Exact settings for this run.                                                  |
| prepared_metadata.json     | Preparation provenance and split/file fingerprints.                           |
| training_audit.json        | Original/synthetic counts and fractional-indicator audit.                     |
| reload_verification.json   | Score agreement before/after serialization on up to 128 training rows.        |
| metadata.json              | Status, schema, dataset/split IDs, versions, hashes, resources, elapsed time. |

These are baseline artifacts, not final thesis evaluation results. Incomplete
or failed bundles are rejected. Abrupt process termination may leave a run marked
running, which loaders also reject. Only load trusted local bundles: joblib uses
pickle and hashes do not authenticate a maliciously replaced bundle.

Python inference example, after a successful training run:

```python
from pathlib import Path
import pandas as pd
from integritree.ml.artifacts import load_bundle
from integritree.ml.inference import predict_records

bundle = load_bundle(Path("artifacts/paysim_phase3_baseline_20260920"))
# Synthetic structured input; not a validated GCash receipt mapping.
inputs = pd.DataFrame([{
    "step": 1, "type": "TRANSFER", "amount": 100.0,
    "nameOrig": "C_EXAMPLE_SENDER", "nameDest": "C_EXAMPLE_RECIPIENT",
}])
results = predict_records(bundle, inputs, ["demo-1"])
```

Use exactly the five raw predictor attributes. Labels, balances, flags and IDs
are not model inputs. Results preserve input order with transaction_id, run_id,
and each model's risk_score and predicted_label. Frontend prediction routes and
batch-export commands are not implemented yet.

The approved four-candidate RF search and common-cutoff search are implemented
in Phase 4; see [METHODOLOGY.md](../docs/METHODOLOGY.md). PR-AUC uses Average
Precision. Original saved configurations remain unchanged.
Freeze selection before official test evaluation.

Verification: **123 tests passed** during Phase 3 implementation, with two existing
third-party deprecation warnings and no broken dependencies. Full PaySim training
and subsequent artifact/inference verification now passed too; no code changes
were needed for the retry. The full-run report is
reports/paysim_phase3_baseline_20260920/verification.json.

The audit confirms 5,076,956 synthetic fraud records and 10,167,052 total RF-SMOTE
training rows (5,083,526 per class). No fractional categorical indicators arose in
this run; the code still preserves them if generated. Both reload checks had zero
score difference on 128 training rows. Source/prepared file hashes were unchanged,
raw/prepared prediction replay agreed on 128 training rows, and individual/batch
predictions agreed on 16 rows. These checks are not performance evaluation.
At the Phase 3 handoff, validation tuning, test evaluation, and SHAP were
unexecuted. Current Phase 4 execution is recorded below.

Use [THESIS_DOCUMENT_CHANGES.md](../docs/THESIS_DOCUMENT_CHANGES.md) for manuscript updates.


## Phase 4 research commands

Run from `backend`. Commands create new output directories; run IDs cannot overwrite
existing results. Full [method and verification notes](../docs/PHASE4_IMPLEMENTATION.md)
explain coverage, edge cases, selected bundles, and SHAP compatibility.

Baseline validation (already executed as `paysim_phase4_baseline_validation_20260923`):

```powershell
& ./.venv/Scripts/python.exe scripts/evaluate_models.py --models artifacts/paysim_phase3_baseline_20260920 --prepared data/prepared/paysim_phase2_20260918 --split validation
```

Run the approved four-candidate validation search when conducting final selection.
This trains three additional model pairs and can take many hours on this machine;
it is not required just to load or inspect the completed baseline validation report.
The previous baseline training alone took about 2 hours 40 minutes.

```powershell
& ./.venv/Scripts/python.exe scripts/select_models.py --baseline artifacts/paysim_phase3_baseline_20260920 --prepared data/prepared/paysim_phase2_20260918 --run-id paysim_validation_selection --jobs 1
```

Resume an interrupted search with the same baseline/prepared arguments and
`--resume artifacts/paysim_validation_selection`. Complete candidates and reports
are reused after provenance checks. Do not change the frozen search plan mid-run.
Only after that selection completes may its bundle be used for the official test:

```powershell
& ./.venv/Scripts/python.exe scripts/evaluate_models.py --models artifacts/paysim_validation_selection --prepared data/prepared/paysim_phase2_20260918 --split test
```

Generate explanations from an existing evaluation report. This preview uses five
shared records; remove `--limit 5` and change `--scope preview` to `--scope sample`
for the approved 1,000-record global sample. Use `--scope full` explicitly for all
report records. Sampling is uniform, seeded, and not stratified by label.

```powershell
& ./.venv/Scripts/python.exe scripts/explain_models.py --models artifacts/paysim_phase3_baseline_20260920 --prepared data/prepared/paysim_phase2_20260918 --report reports/paysim_phase4_baseline_validation_20260923 --scope preview --limit 5
```

SHAP uses the shared original-training background in
`artifacts/explanation_cache/background_<hash>/`; per-record SVG/JSON cache entries
are under `artifacts/explanation_cache/records/`. Reports record actual coverage.
On this machine, five preview records took roughly 0.1?0.2 seconds each for RF
and 0.7?1.0 seconds each for RF-SMOTE attribution alone, excluding initialization,
plotting, and I/O. These measurements are not a latency guarantee for API requests.
No full-dataset SHAP job or final test evaluation has been run.

## Validation-only 1:3 SMOTE sensitivity study

`configs/experiment_smote_1_to_3.yaml` keeps the prepared data, seeds, features,
ordinary SMOTE method, and RF settings fixed while changing the post-resampling
fraud-to-legitimate ratio to 1:3. The 200-tree/depth-20 setting is inherited from
the completed 1:1 validation search. This is a post-validation descriptive study,
not a new predeclared selection procedure.

First resolve the existing 200/depth-20 candidate bundle referenced by
`artifacts/paysim_validation_selection/selection.json`, then train the 1:3 branch
while reusing its verified benchmark RF:

```powershell
& ./.venv/Scripts/python.exe scripts/train_models.py --prepared data/prepared/paysim_phase2_20260918 --config configs/experiment_smote_1_to_3.yaml --reuse-rf-from <existing-200-depth-20-candidate> --jobs 1
```

Evaluate the resulting bundle on validation only, then compare its report with the
existing 1:1 200/depth-20 validation report:

```powershell
& ./.venv/Scripts/python.exe scripts/evaluate_models.py --models <new-1-to-3-bundle> --prepared data/prepared/paysim_phase2_20260918 --config configs/experiment_smote_1_to_3.yaml --split validation
& ./.venv/Scripts/python.exe scripts/compare_smote_ratios.py --one-to-one-report <existing-1-to-1-validation-report> --one-to-three-report <new-1-to-3-validation-report>
```

The comparison writes fingerprinted score and threshold tables, raw 1:3-minus-1:1
differences, a readable summary, and SVG figures. It deliberately records no winner
or selected cutoff. Do not run either ratio on the held-out test split without a
separate final-selection decision.

The full 1:3 run completed as `paysim_smote_1_to_3_200_depth20_20260925` with
1,687,938 synthetic fraud rows and 6,778,034 total training rows. Its validation
report is `paysim_smote_1_to_3_validation_20260925`; the verified comparison is
`paysim_smote_ratio_comparison_validation_20260925`. The comparison found PR-AUC
0.335711 for 1:1 and 0.330708 for 1:3. At the .95 reference checkpoint, 1:3 had
precision 0.293948, recall 0.372263, F1 0.328502, and MCC 0.329825. These are
validation-only descriptive findings, not a selected ratio or final test result.
