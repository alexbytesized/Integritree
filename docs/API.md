# Backend interface contracts

Last updated: 2026-10-01. Researcher CSV analysis and the initial receipt workflows are connected to the frontend. See [receipt routes and wire contracts](RECEIPT_APPLICATION.md). The approved five-category receipt scope is documented in [receipt workflow decisions](RECEIPT_WORKFLOW.md).

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
session directory and temporary receipt storage. It does not load models or datasets, train, or verify inference readiness. Unresolved draft methodology fields
are allowed; malformed configuration stops startup with an error.

Run the app using the factory command in [the backend README](../backend/README.md).
FastAPI provides `/docs`, `/redoc`, and `/openapi.json`.
Research routes are also registered; models load lazily on the first analysis.

CORS permits GET, POST, and DELETE requests with Content-Type, X-Research-Session and X-Receipt-Session headers from `http://localhost:5173` and
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
| `GET /analyses/{id}/records/{row}/waterfall/{model}` | Authenticated signed contribution bar-chart SVG for `rf` or `rf_smote` after computation. |
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
The raw PaySim single-record prediction envelope remains reserved. Receipt endpoints
use separate [confirmed-image contracts](RECEIPT_APPLICATION.md); researcher routes are above.

| Contract | Meaning |
| --- | --- |
| `PaySimRecord` | All 11 raw columns after numeric text has been parsed. |
| `PredictorInput` | Five source attributes: step, type, amount, origin ID, destination ID. Rejects target, flags, balance fields, and unknown extra fields. |
| `SourceRecordIdentity` | Dataset SHA-256 plus one-based source data-row number. Its transaction ID is `<sha256>:<row>`. |
| `LabeledTransaction` | Identity and predictor input with a separate actual label. |
| `ModelResult` | Model name, class label, 0..1 risk score, optional explanation. |
| `PairedPrediction` | Transaction ID, run ID, and separately identified RF and RF-SMOTE results. |
| `ShapExplanation` | Output space, baseline, explained output, and uniquely named feature contributions. |
| `ExperimentMetadata` | Run/time/dataset identity, configuration and split fingerprints, unique feature order, and package versions. |

Numeric contract fields must be finite. Amounts and balances must be nonnegative;
zero remains valid. Labels are integer 0 or 1; steps and source rows are positive
integers. Transaction type must match one of the five PaySim categories.
Phase 2 implements CSV parsing and an ingestion boundary for permitted missing
amount/type values. Complete-record HTTP contracts remain strict; receipt mapping
uses the separate implemented receipt adapter.

Paired results allow model disagreement and contain no fabricated correctness
or evaluation metrics. Evaluation must join separately established ground truth
using preserved transaction identity. The current SHAP contract validates shape,
not numerical additivity or consistency with the implemented model output.

These contracts define data boundaries; they do not prove that arbitrary labeled
datasets or real GCash transactions are compatible with a PaySim-trained model.
The engineered feature list and simulation-time convention are now approved in
[METHODOLOGY.md](METHODOLOGY.md). Scores now use mean-tree fraud probability with
the frozen shared >=0.43 cutoff for the current selected researcher bundle. Receipt mapping and job contracts are documented in the receipt application guide.

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

This raw PaySim helper is not a standalone HTTP route. Python callers use `load_bundle(path)`
and `predict_records(bundle, inputs, transaction_ids)`. Inputs contain exactly the
five raw predictor attributes and reuse saved preprocessing without refitting.
The returned DataFrame preserves input order and includes `transaction_id`,
`run_id`, `rf_risk_score`, `rf_predicted_label`, `rf_smote_risk_score`, and
`rf_smote_predicted_label`. IDs must be nonempty strings and unique within a batch.
Ground truth is supplied separately to evaluation, never to the models.
Scores are 0..1; SHAP explanations and correctness fields are not fabricated.

## Approved application contract requirements

The following are cross-phase requirements. The researcher routes above and
[receipt application guide](RECEIPT_APPLICATION.md) define implemented HTTP schemas;
requirements below are not literal wire field names. See [the alignment audit](BACKEND_ALIGNMENT_AUDIT.md) and
[the phase plan](BACKEND_IMPLEMENTATION_PLAN.md).

| Response area | Required semantics |
| --- | --- |
| Analysis identity | Stable internal analysis ID, input revision, record identity, and trusted model/explainer provenance. Keep receipt reference separate; preserve leading zeros. |
| Record details | Original Inputs as supplied/confirmed; Derived Inputs with all eleven engineered features and the actual scaled matrix/feature order. Ground truth is nullable and separately joined. No separate correctness badge is required. |
| Paired model result | Explicit rf/rf_smote keys, class 0/1, underlying fraud score 0..1, saved common threshold, and band. Presentation uses Predicted Fraud / Predicted Legitimate and score 100*p. |
| Bands | [0,20) Minimal Risk; [20,40) Low Risk; [40,60) Moderate Risk; [60,80) High Risk; [80,100] Critical Risk. Class and band use unrounded values and separate rules. |
| Explanation | Per-model state, top positive contributor or explicit no-positive state, deterministic narrative, contribution bar-chart asset and accessible description. Pending/failed is different from no positive contribution. Keep numerical contributions internally; a frontend numerical table is suggestion-only. |
| Evaluation | Independently labeled, declared population; metric value or null/status/reason, named PR-AUC method, confusion matrices, comparison method/units, paired McNemar table/method/statistic/p-value/alpha/status. Actual methods drive help copy. |
| Export | `integritree_results.zip` contains exactly `Experiment-Paper_YYYY-MM-DD.pdf` and `Raw-Data_YYYY-MM-DD.csv`, using one Philippine export-start date. Complete-upload CSV data and export-status metadata are preserved; JSON/HTML attachments are omitted. |
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

The current PredictorInput request remains a raw PaySim contract. Separate internal
[confirmed-receipt contracts and paired inference](RECEIPT_MAPPING.md) now validate
optional reference string, PHP principal, Manila
date/time, original GCash workflow label, mapped PaySim category, account roles,
optional masked names, and extraction/correction provenance. The receipt HTTP
service creates server-owned source context from a validated image job; it never
accepts that context from a client request. The initial adapter accepts Express Send, Pay Online and confirmed
bank-account transfers; QR and cash categories are rejected until supported. Target Express Send
(TRANSFER), over-the-counter cash-in (CASH_IN), over-the-counter cash-out (CASH_OUT),
wallet-funded merchant QR or Pay Online (PAYMENT), and GCash-to-bank transfer (DEBIT),
using app screenshots only. For confirmed personal-wallet origins, merchant flags
are 0/0 for personal-wallet TRANSFER, 0/1 for merchant PAYMENT, and 0/0 for bank-account
DEBIT; retain actual role labels separately. Cash-agent mappings remain pending.
Reject unresolved required roles or unsupported workflows before prediction.
Permit missing references, preserve leading zeros when present, and use a separate
internal analysis ID. Full names are not required predictors. Permit correction and
completion after image upload, followed by explicit confirmation; no image-free
manual workflow. Return field-level validation and unsupported-layout/type states.

The internal adapter derives the eleven unscaled features under `gcash_confirmed_v1`
and reuses saved scaling once through `transform_engineered`. Do not synthesize a step or a fake PaySim account ID.
Do not treat supplied derived values as automatically validated model inputs.
Immutable confirmations retain currency/calendar assumptions and mapping version.
Revision fingerprints and `is_current` enable stale-result detection; the service
invalidates cached results when confirmed inputs change. No receipt-derived ground truth or accuracy metrics.

The local [OCR baseline](RECEIPT_OCR_BENCHMARK.md) provisionally selects RapidOCR.
Its parser returns candidates only; it does not validate account roles or authorize
prediction. A Bank Transfer heading can also lead to a wallet destination.
Accept one PNG/JPEG at a time, at most 10 MiB and 20,000,000 decoded pixels. Release
verified workflows first; CASH_IN/CASH_OUT await samples. Application images, confirmed
details, and results are temporary until Clear or backend shutdown/restart.
Session access isolation and cleanup are implemented separately from retained
development screenshots in `data/raw/receipt_samples/`. Individual ZIPs contain only the four-page
`Transaction-Results_YYYY-MM-DD.pdf`; internal receipt data remains available to
the application. See the receipt application guide
for exact routes, states, errors and ZIP contracts.

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

These Phase 4 Python interfaces now underpin researcher and receipt HTTP services:

- `ml.research.evaluate_run(...)`: paired full-split predictions, metrics, AP,
  comparisons, McNemar, figures, and provenance. Validation is the default;
  `split="test"` requires a frozen validation-selected bundle.
- `ml.staged_selection.select_three_stage(...)`: sequential ratio/AP, forest/AP,
  threshold/F1 selection on exactly 1%, 2%, ..., 100%, with frozen stage evidence and resume support.
  `stop_after_stage=1` freezes the ratio, 2 freezes the forest, and default 3
  executes all stages. Graceful pauses return a run with `status: paused`.
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

These are local execution interfaces wrapped by application services with bounded
work queues, status/error responses, ownership, portable asset URLs/downloads,
and researcher pagination. Never expose a local filesystem
path directly as the final client asset contract. No new unlabeled request should
receive evaluation metrics or an invented actual label. See
[Phase 4 contracts and evidence](PHASE4_IMPLEMENTATION.md).

### SHAP chart presentation (2026-10-06)

All current SHAP charts show rounded horizontal contributions in the fixed model feature order, with friendly feature names and all eleven rows included. Negative green (`#009900`) bars decrease the fraud risk score; positive red (`#A00000`) bars increase it. The axis is fixed at -100 to +100 percentage points, with gridlines and ticks every 10. Nonzero labels use three significant digits in percentage points; exact-zero rows have no bar or numerical label. In the modal, Reference Score, Output Score, and Top Risk-Increasing Contributor appear below the graph as full-width sections with blue headings, always-visible explanatory paragraphs, and a bulleted value. Standalone exports retain their summary cards. SHAP values, input values, reconstruction checks, and model predictions are unchanged.

The `waterfall` URL segments and `waterfall_url` / `waterfall_path` fields remain legacy compatibility names and now identify contribution bar-chart SVGs. Both chart endpoints accept `layout=modal|standalone` (default `standalone`). The modal SVG contains only the plotting area, endpoint padding, and ticks; React supplies the heading, stationary feature labels, direction captions, accessible explanatory sections and bulleted summary values. Standalone charts retain model/transaction identification and equivalent summary cards. Chart caches include layout, contributor metadata, a presentation version, and a fingerprint of stored values and labels; older assets are regenerated without recomputing SHAP. Receipt ZIPs contain only `Transaction-Results_YYYY-MM-DD.pdf`; SVGs remain available through chart endpoints and are not rendered for inclusion in downloads.


The shared geometry contract is `backend/src/integritree/ml/shap_geometry.json`, packaged with the backend and imported directly by React. SVG coordinates are displayed at 100 CSS pixels per figure inch: 80px per 10 points, a 1600px plotting area, 100px padding at each endpoint, 8px top padding, 40px rows, and a 40px tick strip. Thus an eleven-row modal SVG is displayed at 1800 x 488px; row centers are at y=28+40*row and zero is at x=900. Feature names and numerical values have normal font weight. Tiny nonzero bars retain their true length and receive a visible hairline marker; exact-zero rows remain blank. Standalone exports show the complete axis, identification, and summary cards.

The SHAP modal uses a maximum width of 1024px to fit the stationary labels and plotting viewport without a second horizontal scrollbar on desktop. It otherwise inherits McNemar's shared modal styles: maximum height 620px subject to viewport limits, the same inner padding and responsive adjustments, and 80% scaling on desktops at least 1200px wide. These are CSS dimensions before shared desktop scaling. Content scrolls vertically within the modal. The independent plot viewport remains 680-760px wide in CSS coordinates, centered on zero after each image load so exactly nine gridlines (-40 through +40) initially appear within that viewport. Ordinary rerenders preserve the user's scroll position. The fixed 208px feature column uses the same 40px rows, with an 8px gap before the plotting viewport. The chart is introduced by the blue section heading `SHAP Feature Contributions`; the old centered chart title is omitted in the modal. The surrounding chart section scrolls horizontally wherever the full chart layout exceeds the modal content width, on narrower screens. Summary sections fill the available content width and need no information buttons. Standalone SVGs and researcher reports retain their existing presentation. Receipt PDFs include interleaved SHAP evaluation pages with nine adaptive gridlines and summary cards; see [receipt downloads](RECEIPT_APPLICATION.md#download-and-retention). Cache version `contribution_bars_v3`, SHAP calculations, API fields, receipt revision/ownership checks, and export filenames are preserved.
