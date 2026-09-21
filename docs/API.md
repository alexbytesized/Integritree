# Backend interface contracts

Last updated: 2026-09-22. HTTP remains health-only; local preparation, baseline training, and Python inference are implemented.

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
not numerical additivity or consistency with the implemented model output.

These contracts define data boundaries; they do not prove that arbitrary labeled
datasets or real GCash transactions are compatible with a PaySim-trained model.
The engineered feature list and simulation-time convention are now approved in
[METHODOLOGY.md](METHODOLOGY.md). Scores now use mean-tree fraud probability with
a shared >=0.50 baseline cutoff. Receipt mapping and prediction endpoint contracts
remain decisions for their dependent phases.

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
unapproved settings, and output conflicts fail with exit 2. Baseline training is
implemented via `python scripts/train_models.py --prepared <bundle>`; see the
backend README. Evaluation and batch-export scripts remain guarded placeholders.

See [the backend README](../backend/README.md#prepare-the-approved-paysim-dataset)
for bundle file schemas, original-row identity, separate labels, integrity checks,
and preprocessing reload. A preparation bundle is usable only when its metadata
status is `complete`; it is not a trained model or an evaluation result.

## Phase 3 Python inference boundary

No prediction HTTP route has been added. Python callers use `load_bundle(path)`
and `predict_records(bundle, inputs, transaction_ids)`. Inputs contain exactly the
five raw predictor attributes and reuse saved preprocessing without refitting.
The returned DataFrame preserves input order and includes `transaction_id`,
`run_id`, `rf_risk_score`, `rf_predicted_label`, `rf_smote_risk_score`, and
`rf_smote_predicted_label`. IDs must be nonempty strings and unique within a batch.
Ground truth is supplied separately to future evaluation logic, never to the models.
Scores are 0..1; SHAP explanations and correctness fields are not fabricated.

## Approved future contract requirements (not available endpoints)

The following are Phase 4-6 implementation requirements, not descriptions of the
current OpenAPI schema. Exact route names and wire schemas must be documented
when implemented. See [the alignment audit](BACKEND_ALIGNMENT_AUDIT.md) and
[the phase plan](BACKEND_IMPLEMENTATION_PLAN.md).

| Response area | Required semantics |
| --- | --- |
| Analysis identity | Stable internal analysis ID, input revision, record identity, and trusted model/explainer provenance. Keep receipt reference separate; preserve leading zeros. |
| Record details | Original Inputs as supplied/confirmed; Derived Inputs with all eleven engineered features and the actual scaled matrix/feature order. Ground truth is nullable and separately joined. No separate correctness badge is required. |
| Paired model result | Explicit rf/rf_smote keys, class 0/1, underlying fraud score 0..1, saved common threshold, and band. Presentation uses Predicted Fraud / Predicted Legitimate and score 100*p. |
| Bands | [0,20) Minimal Risk; [20,40) Low Risk; [40,60) Moderate Risk; [60,80) High Risk; [80,100] Critical Risk. Class and band use unrounded values and separate rules. |
| Explanation | Per-model state, top positive contributor or explicit no-positive state, deterministic narrative, waterfall asset and accessible description. Pending/failed is different from no positive contribution. Keep numerical contributions internally; a frontend numerical table is suggestion-only. |
| Evaluation | Independently labeled, declared population; metric value or null/status/reason, named PR-AUC method, confusion matrices, comparison method/units, paired McNemar table/method/statistic/p-value/alpha/status. Actual methods drive help copy. |
| Export | ZIP with results.csv, evaluation.json, offline report.html, metadata.json. Include run/evaluation context here, not a required on-screen panel. Optional raw SHAP CSV must declare subset coverage. |
| Job | Bounded asynchronous analysis/explanation/export work, progress/stage, partial/failed/expired states, actionable recovery, and declared download scope/lifetime. |

`run_id` and explanation versions remain provenance, not UI panel requirements.
Color-only visual association does not permit an ambiguous backend model mapping.
Avoid including sensitive raw receipt details in logs or error bodies.

### Research input and query rules

- Publish an exact upload schema and downloadable CSV template. Reuse predictor
  validation and saved preprocessing, not source-fingerprint-bound preparation or
  training. The five raw predictor sources can construct the eleven model features;
  `isFraud` is separately required for evaluation. Balance columns are not predictors.
- Search transaction IDs across the complete analysis before pagination, with
  stable source ordering and total/filtered counts. Risk-score sorting is not required.
- Model filter: Both, RF-SMOTE, Benchmark RF. Prediction outcome: All, TP, FP, TN, FN.
  A non-All outcome requires one model and ground truth. Both resets the UI outcome
  to All; the server rejects contradictory combinations. Table queries must not
  silently change official aggregate evaluation results.
- Keep arbitrary uploaded-data evaluations distinct from official held-out test
  results. No-label inputs have unavailable evaluation, not fabricated labels.

### Receipt boundary

The current PredictorInput request is a raw PaySim contract. Add a separate
confirmed-receipt contract for reference string, amount, date/time, TRANSFER,
sender/recipient types, optional masked names, and extraction/correction provenance.
The client can display all categories but the server accepts only the supported
person-to-person transfer workflow. Unknown/merchant roles are incompatible with
that scope. Missing-reference policy remains to be finalized; full names are not
required predictors. Return field-level validation and unsupported-layout/type states.

Derive the eleven unscaled features under a recorded demonstration mapping and
reuse saved scaling once. Do not synthesize a step or a fake PaySim account ID.
Do not treat supplied derived values as automatically validated model inputs.
Store currency/calendar assumptions and adapter version; invalidate prior results
when confirmed inputs change. No receipt-derived ground truth or accuracy metrics.

### Configuration and artifact compatibility

Active evaluation configuration selects signed_over_mean, with
`100*(S-B)/((S+B)/2)` for nonnegative metrics with positive mean. Both zero gives
null/zero_denominator. Either MCC negative uses `S-B`, labeled coefficient units.
Undefined inputs propagate an unavailable status. Legacy absolute_over_mean remains
readable and must keep its own method label. Calculation is not implemented yet.

Use a separately recorded evaluation/explainer policy referencing immutable model
artifacts. The Phase 3 artifact loader currently enforces fixed-baseline settings;
Phase 4 must explicitly support validation-selected thresholds/runs and test legacy
compatibility before serving them. Do not mutate a saved configuration to pass a gate.

Current GET-only CORS will need the actual methods/headers when upload and prediction
routes are added. A database is optional; file/job access and retention still need
an explicit design. Mock-up file-size labels are not implemented server limits.
