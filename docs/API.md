# Backend interface contracts

Last updated: 2026-09-29. Researcher CSV analysis is connected to the frontend; receipt APIs remain future work.

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
the experiment configuration and creates/cleans its isolated temporary research
session directory. It does not load models or datasets, train, or verify inference readiness. Unresolved draft methodology fields
are allowed; malformed configuration stops startup with an error.

Run the app using the factory command in [the backend README](../backend/README.md).
FastAPI provides `/docs`, `/redoc`, and `/openapi.json`.
Research routes are also registered; models load lazily on the first analysis.

CORS permits GET, POST, and DELETE requests with Content-Type and X-Research-Session headers from `http://localhost:5173` and
`http://127.0.0.1:5173` by default. Configure explicit origins with
`INTEGRITREE_CORS_ORIGINS` as a JSON array in local settings.
CORS is browser configuration, not authentication.

## Researcher HTTP routes

See [workflow, inputs, retention, and capacity](RESEARCHER_WORKFLOW.md).
`POST /api/v1/research/sessions` returns an opaque `token`; send it in
`X-Research-Session` for every analysis request. Tokens are memory-only and
browser sessionStorage preserves them across refresh. Missing/expired sessions
return 410; another session's analysis returns 404.

| Method and path under `/api/v1/research` | Behavior |
| --- | --- |
| `GET /template` | Download a six-column demonstration CSV. |
| `POST /analyses?filename=example.csv` | Stream a raw UTF-8 CSV request body (`Content-Type: text/csv`); return 202 with `id` and status. Not multipart. |
| `GET /analyses/{id}` | Status, processed rows, byte count, errors/issues, frozen threshold, complete evaluation, and export status. |
| `GET /analyses/{id}/records` | Ten-row pages; `page`, `search`, `model=both\|rf\|rf_smote`, `outcome=all\|tp\|fp\|tn\|fn`. Outcome requires a single model. |
| `GET /analyses/{id}/records/{row}` | One-based upload record, original fields, unscaled derived fields, model inputs, paired scores/labels, and explanation state. |
| `POST /analyses/{id}/records/{row}/explanation` | Queue paired on-demand SHAP; 202. Duplicate pending/computed requests reuse work; failed requests retry. |
| `GET /analyses/{id}/records/{row}/waterfall/{model}` | Authenticated SVG for `rf` or `rf_smote` after computation. |
| `POST /analyses/{id}/exports` | Queue complete-upload ZIP; 202. Repeat after completion refreshes explanation coverage. |
| `GET /analyses/{id}/exports/download` | Download completed ZIP. |
| `DELETE /analyses/{id}` | Clear an inactive complete/failed analysis; 204. Active work returns 409. |

Analysis states: `uploading`, `queued`, `loading_models`, `processing`,
`evaluating`, `complete`, `failed`. Explanation states: `not_requested`, `pending`,
`computed`, `failed`; computed model contributions can report `no_positive_contributor`.
Export states: `not_requested`, `pending`, `complete`, `failed`.

Records and metrics are unavailable until the entire input passes. Upload errors
include HTTP 413 for actual bytes over 500 MiB, 429 for a full bounded queue, and
400/415 for invalid request format. CSV validation failures are asynchronous job
failures with `error` and row/column `issues`; retry by uploading a corrected file.
No record-count cap or official-test selector/check exists. Export metadata uses
`uploaded_dataset` scope and explicitly declares membership unverified. Optional
PaySim balances/flag are source information, not model features.

## Shared contracts for later phases

Definitions live in `backend/src/integritree/contracts.py`; the reserved request
envelope is in `backend/src/integritree/api/schemas.py`.
The single-record prediction and receipt endpoints remain reserved. Research endpoints are documented below.

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
the frozen shared >=0.43 cutoff for the current selected researcher bundle. Receipt mapping and prediction endpoint contracts
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
backend README. Local evaluation and the researcher HTTP upload/export flow are implemented.

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
Ground truth is supplied separately to evaluation, never to the models.
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
readable and must keep its own method label. Calculation is implemented and the
researcher disk adapter is tested against the existing evaluator.

Use a separately recorded evaluation/explainer policy referencing immutable model
artifacts. Ordinary fitted bundles retain their integrity checks; completed
three-stage selections use schema 2 and carry the frozen common threshold.
Do not mutate a saved configuration to pass a gate.

Research CORS, session ownership, temporary SQLite storage, and the 500 MiB
server limit are implemented. See the researcher route table and workflow above.


## Phase 4 Python interfaces (2026-09-23)

These Phase 4 Python interfaces now underpin the researcher HTTP services. The
single-record receipt routes remain future work:

- `ml.research.evaluate_run(...)`: paired full-split predictions, metrics, AP,
  comparisons, McNemar, figures, and provenance. Validation is the default;
  `split="test"` requires a frozen validation-selected bundle.
- `ml.staged_selection.select_three_stage(...)`: sequential ratio/AP, forest/AP,
  threshold/F1 selection on exactly 1%, 2%, ..., 100%, with frozen stage evidence and resume support.
  `stop_after_stage=1` freezes the ratio, 2 freezes the forest, and default 3
  executes all stages. Graceful pauses return a run with `status: paused`.
- `ml.stage3_reset.reset_stage3(run, backend, protocol)`: explicitly clears Stage 3
  outputs, preserves Stage 1-2 candidate evidence, and regenerates review snapshots;
  its journal supports interrupted-reset recovery.
- `ml.staged_selection.request_pause(path)`: asks an active selection worker to
  stop after its current candidate checkpoint; returns `pause_requested`.
- `ml.staged_selection.export_forest_stage(...)`: verifies and exports a completed
  Stage 2 review without threshold selection or test access.
- `ml.artifacts.load_bundle(path)`: loads an ordinary fitted pair or a complete schema-2
  three-stage selection after integrity checks; obsolete selections are rejected.
- `ml.explainability.ExplanationEngine(...).explain(features, transaction_ids,
  analysis_id)`: both-model explanations from already-prepared features in saved
  order. Each computed item includes base/output probability, contributions,
  readable values, top-positive status, deterministic narrative, SVG path,
  reconstruction error, and cache/provenance identity.
- `ml.explainability.explain_report(...)`: paired preview/sample/full coverage from
  an existing evaluation report, selected identities, and mean absolute SHAP summary.

These are local execution interfaces, not yet thread-safe HTTP job contracts.
Phase 5 must define bounded work queues, status/error responses, ownership,
portable asset URLs/downloads, and pagination. Never expose a local filesystem
path directly as the final client asset contract. No new unlabeled request should
receive evaluation metrics or an invented actual label. See
[Phase 4 contracts and evidence](PHASE4_IMPLEMENTATION.md).
