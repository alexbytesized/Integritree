# Backend interface contracts

Last updated: 2026-09-18. HTTP interface remains health-only; Phase 2 adds local preparation commands.

## Available HTTP endpoint

`GET /api/v1/health` returns HTTP 200:

```json
{
  "status": "ok",
  "service": "integritree-backend",
  "version": "0.1.0"
}
```

This confirms application availability. Startup validates application settings and
the experiment draft, but does not load the dataset, create output directories,
train models, or verify inference readiness. Unresolved draft methodology fields
are allowed; malformed configuration stops startup with an error.

Run the app using the factory command in [the backend README](../backend/README.md).
FastAPI provides `/docs`, `/redoc`, and `/openapi.json`.
Only the health route is registered as an application endpoint.

CORS permits GET requests from `http://localhost:5173` and
`http://127.0.0.1:5173` by default. Configure explicit origins with
`INTEGRITREE_CORS_ORIGINS` as a JSON array in local settings.
CORS is browser configuration, not authentication.

## Shared contracts for later phases

Definitions live in `backend/src/integritree/contracts.py`; the reserved request
envelope is in `backend/src/integritree/api/schemas.py`.
There is no prediction, receipt, upload, or research endpoint yet.

| Contract | Meaning |
| --- | --- |
| `PaySimRecord` | All 11 raw columns after numeric text has been parsed. |
| `PredictorInput` | Five source attributes: step, type, amount, origin ID, destination ID. Rejects target, flags, balance fields, and unknown extra fields. |
| `SourceRecordIdentity` | Dataset SHA-256 plus one-based source data-row number. Its transaction ID is `<sha256>:<row>`. |
| `LabeledTransaction` | Identity and predictor input with a separate actual label. |
| `PredictionRequest` | Transaction ID and predictor input, without ground truth. Reserved for a future route. |
| `ModelResult` | Model name, class label, 0..1 risk score, optional explanation. |
| `PairedPrediction` | Transaction ID, run ID, and separately identified RF and RF-SMOTE results. |
| `ShapExplanation` | Output space, baseline, explained output, and uniquely named feature contributions. |
| `ExperimentMetadata` | Run/time/dataset identity, configuration and split fingerprints, unique feature order, and package versions. |

Numeric contract fields must be finite. Amounts and balances must be nonnegative;
zero remains valid. Labels are integer 0 or 1; steps and source rows are positive
integers. Transaction type must match one of the five PaySim categories.
Phase 2 implements CSV parsing and an ingestion boundary for permitted missing
amount/type values. Complete-record HTTP contracts remain strict; receipt mapping
is a later responsibility.

Paired results allow model disagreement and contain no fabricated correctness
or evaluation metrics. Evaluation must join separately established ground truth
using preserved transaction identity. The current SHAP contract validates shape,
not numerical additivity or consistency with an as-yet-unimplemented model.

These contracts define data boundaries; they do not prove that arbitrary labeled
datasets or real GCash transactions are compatible with a PaySim-trained model.
The engineered feature list and simulation-time convention are now approved in
[METHODOLOGY.md](METHODOLOGY.md). Score semantics, receipt-to-simulation mapping,
and endpoint contracts remain decisions for their dependent phases.

## Configuration command

`python -m integritree.config` validates the draft and returns JSON with
`valid`, `experiment_name`, `unresolved`, and an explanatory note; exit 0.
An invalid configuration returns `valid: false` and an `error`; exit 2.

`--require-stage prepare|train|evaluate|explain` also rejects unresolved settings
for that stage and preceding stages. This checks configuration readiness only.
It does not perform research or authorize a method change.

The preparation readiness check now passes for the approved configuration.
`python scripts/prepare_data.py` runs Phase 2 and returns success JSON containing
the completed bundle path (exit 0). Progress is written to stderr; invalid data,
unapproved settings, and output conflicts fail with exit 2. Training, evaluation,
and batch prediction scripts remain guarded placeholders.

See [the backend README](../backend/README.md#prepare-the-approved-paysim-dataset)
for bundle file schemas, original-row identity, separate labels, integrity checks,
and preprocessing reload. A preparation bundle is usable only when its metadata
status is `complete`; it is not a trained model or an evaluation result.
