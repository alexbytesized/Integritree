# Integritree backend

Phases 1 and 2 are complete and provide packaging, settings, validated configuration, shared
contracts, a health endpoint, and batched PaySim preparation. Model training,
prediction, evaluation, and OCR remain later-phase work.

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
& .\.venv\Scripts\python.exe -m piptools compile --extra dev --generate-hashes --allow-unsafe --strip-extras --no-emit-index-url --output-file requirements-dev.lock pyproject.toml
```

The lock was generated on Windows with Python 3.12; installation on other
platforms has not been verified. Phase 2 adds NumPy, pandas, PyArrow, and scikit-learn. SMOTE/SHAP and OCR
libraries will be added in their respective phases.

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

```powershell
& .\.venv\Scripts\python.exe -m integritree.config
& .\.venv\Scripts\python.exe -m integritree.config --require-stage prepare
```

The first command validates the draft and lists unresolved fields (exit 0).
The second now passes (exit 0): the researchers approved the Phase 2 preparation
settings. Training and later stages still fail readiness checks until their
remaining choices are settled. Available stages are `prepare`, `train`,
`evaluate`, and `explain`; requirements include earlier stages.
`--config` accepts an absolute path or one relative to the backend root.

Validation does not approve a methodology or execute research. `scripts/prepare_data.py` now executes Phase 2 as documented below. The other
three scripts remain guarded placeholders (exit 2). No command produces mock
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

| Bundle file | Contents |
| --- | --- |
| `metadata.json` | Completion status, dataset/run identity, split counts, package versions, code/file hashes. |
| `configuration.json` | Exact validated experiment settings used for preparation. |
| `audit.json` | Source hash, scanned/retained counts, missingness, numeric ranges, class/type counts, duplicate count, failures. |
| `source.parquet` | Retained typed source columns plus original `source_row_number`; includes categorical data for later SMOTE decisions. |
| `duplicates.parquet` | Removed original row numbers and the first matching row number. |
| `split_manifest.parquet` | Original row number, split name, and separate `actual_label`. |
| `preprocessing.json` | Training amount median, scaler coefficients/extrema, feature order, and training count. |
| `train_features.parquet` | Training model features plus row identity; no labels/balances/raw source features. |
| `validation_features.parquet` | Validation features with the same schema and transformations. |
| `test_features.parquet` | Test features with the same schema and transformations. |

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

| Location | Responsibility |
| --- | --- |
| `configs/` | Experiment configuration and seeds. |
| `scripts/` | Working preparation command; other commands remain guarded placeholders. |
| `src/integritree/settings.py` | Local paths and environment settings. |
| `src/integritree/config.py` | Experiment schema, validation, and stage requirements. |
| `src/integritree/contracts.py` | Shared transaction/result/provenance contracts. |
| `src/integritree/api/` | HTTP interfaces; currently health only. |
| `src/integritree/services/` | Future workflow coordination. |
| `src/integritree/ml/` | Working shared data/features/preprocessing/preparation; later ML modules remain placeholders. |
| `src/integritree/receipts/` | Future GCash extraction and confirmed-input mapping. |
| `tests/` | Synthetic foundation and preparation tests; later-phase placeholder files remain. |
| `data/raw/` | Original local PaySim CSV; preserve the source. |
| `data/prepared/<run_id>/` | Local preparation bundles, audits, and split membership. |
| `artifacts/<run_id>/` | Future models, preprocessing, and provenance bundles. |
| `reports/<run_id>/` | Future predictions, metrics, statistics, and figures. |
| `runtime/` | Future temporary uploads and exports. |

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
