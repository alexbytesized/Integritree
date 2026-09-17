# Integritree backend

Status: directory scaffold only. Phase 1 code implementation has not started.

Read [the thesis context](../docs/THESIS_CONTEXT.md) and
[the phased implementation plan](../docs/BACKEND_IMPLEMENTATION_PLAN.md)
before implementing functionality.

## What exists now

Python files contain descriptive docstrings only. The configuration files
contain comments only. No dependencies have been installed, no server is
available, and no training, prediction, OCR, or evaluation code is implemented.

The test files are placeholders, not a passing test suite. Running the script
placeholders performs no operations and does not produce experiment results.
Package installation and run commands will be documented in Phase 1.

## Directory responsibilities

| Location | Purpose |
| --- | --- |
| `configs/` | Versioned experiment settings and reproducibility parameters. |
| `scripts/` | Research commands that call the shared Python package. |
| `src/integritree/api/` | Frontend requests, response contracts, and routes. |
| `src/integritree/services/` | Coordinate prediction, research results, and exports. |
| `src/integritree/ml/` | Shared data preparation, training, inference, SHAP, and evaluation. |
| `src/integritree/receipts/` | OCR, supported GCash field extraction, and input mapping. |
| `tests/` | Meaningful tests and small synthetic fixtures as phases are implemented. |
| `data/raw/` | Original local PaySim dataset; preserve the source file. |
| `data/prepared/` | Prepared records, split membership, and data audit outputs. |
| `artifacts/` | Model and fitted-preprocessing bundles, grouped by experiment run. |
| `reports/` | Transaction predictions, metrics, statistical results, and figures. |
| `runtime/uploads/` | Temporary application uploads. |
| `runtime/exports/` | Temporary user downloads. |

Each Python package directory contains `__init__.py`.
The `src/` layout separates reusable code from commands and generated files.

## Shared logic

Research scripts and application services must import the same functions from
`integritree.ml`. The frontend and receipt parser must not recreate feature
engineering or scoring independently.

Training runs through research scripts. Application prediction loads saved
models and fitted preprocessing; it does not retrain or refit on uploads.

The initial individual demonstration supports GCash person-to-person transfer
receipts and displays both RF and RF-SMOTE results. It assesses transaction
behavior, not receipt authenticity.

## Local data and experiment outputs

The backend `.gitignore` excludes raw/prepared datasets, trained artifacts,
generated reports, temporary uploads/exports, environments, and secrets.
Directory markers are retained so empty folders survive a checkout.

Use one run identifier to connect the model artifacts and reports. A completed
run should retain its configuration, dataset fingerprint, split provenance,
feature schema, package versions, and both models' outputs.

Commit source code, reviewed configurations, and small synthetic fixtures.
Do not place real personal receipts in test fixtures.

## Configuration status

`pyproject.toml`, `.env.example`, and `configs/experiment.yaml` are placeholders.
Do not treat sample values from earlier conversations as approved hyperparameters.

FastAPI remains the recommended web framework, pending the Phase 1 dependency
and interface review. No framework has been installed or implemented.

## Technical references

- [Python project configuration](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/)
- [FastAPI application structure](https://fastapi.tiangolo.com/tutorial/bigger-applications/)
