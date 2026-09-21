# Integritree backend implementation plan

Last updated: 2026-09-22 (completed mock-up review and backend alignment audit)

## Current status and scope

The backend scaffold and Phases 1-3 are implemented. Full PaySim preparation and
baseline RF/RF-SMOTE training are complete and verified. Both saved models are in
backend/artifacts/paysim_phase3_baseline_20260920/. The initial RAM preflight stop
was resolved on retry. No validation tuning or official test evaluation has run.
Evaluation, tuning, SHAP, prediction routes, and receipt workflows remain future work.
See [setup instructions](../backend/README.md), [methodology decisions](METHODOLOGY.md),
and [API contracts](API.md).

The mock-up corrections are established in [the decision record](MOCKUP_EVALUATION_DECISIONS.md).
The latest SHAP scope is top positive contributor, deterministic narrative, and
waterfall-only expanded details. A numerical table is a suggestion, not a phase
deliverable. [The code alignment audit](BACKEND_ALIGNMENT_AUDIT.md) records evidence,
the small signed-comparison configuration fix, and remaining implementation gaps.
The audit passed 125 tests; it did not rerun full training or produce thesis metrics.

The planned application will support two uses of the same saved RF and RF-SMOTE models:

- Research: labeled PaySim test records, predictions, explanations, metrics,
  statistical comparisons, and exports.
- Individual demonstration: GCash person-to-person transfer receipts, confirmed
  transaction details, and both models' predictions, scores, and explanations.

Read [THESIS_CONTEXT.md](THESIS_CONTEXT.md) for the manuscript requirements,
confirmed scope, and corrections discussed with the researchers. Scaffolding
does not approve an algorithm change or a Chapter 3 rewrite.

## Phase overview

| Phase | Focus | Main outcome | Status |
| --- | --- | --- | --- |
| 0 | Directory scaffold | Clearly labeled module/configuration placeholders | Complete |
| 1 | Foundation and contracts | Reproducible development setup and validated configuration/data contracts | Complete |
| 2 | Dataset and preprocessing | Audited, reproducible splits and training-fitted transformations | Complete |
| 3 | Training and saved inference | Baseline trainer and reloadable paired model bundles | Complete; full PaySim baseline verified |
| 4 | Research evaluation and SHAP | Verified evaluation reports and model explanations | Not started |
| 5 | Application services and API | Research and individual structured-record workflows | Not started |
| 6 | GCash receipt demonstration | Image extraction, confirmation, mapping, and prediction | Not started |
| 7 | Integration and reproducibility | Backend handoff validated against the frontend workflows | Not started |

The existing foundation is reusable; Phases 1-3 are complete for their stated
PaySim/baseline scope. Their current contracts are not the complete application
contract. Extend them in the dependent phases rather than retrofitting receipt
assumptions into PaySim preparation. Current HTTP is health-only.

### Decisions that every future phase must preserve

- Internal fraud scores remain 0..1. Display score is 100*p; use unrounded values
  for communication bands: [0,20) Minimal, [20,40) Low, [40,60) Moderate, [60,80)
  High, [80,100] Critical. Classification uses the saved shared threshold, not
  band boundaries. UI labels are Predicted Fraud / Predicted Legitimate.
- Both models must stay joined to the same record. Ground truth is separate and
  optional outside evaluation. No individual-detail correctness badge is required.
- Original Inputs and Derived Inputs are required in record details. Distinguish
  readable engineered values from the scaled matrix actually consumed by the model.
- Model/run and evaluation context belong in internal provenance and the approved
  ZIP export, not new on-screen model-run or evaluation-detail panels.
- Retain color-only visible model association in table score/contributor cells;
  backend values must still carry explicit model keys. Visible badges/subheaders
  and optional tooltips remain suggestions. No risk-score sorting.
- Numerical SHAP table, standalone raw-contribution download, global SHAP plots,
  PDF/XLSX export, and agreement/disagreement filters are optional suggestions.
- Receipt support is GCash person-to-person only. All five types may appear in
  the dropdown, but only TRANSFER is enabled. Server validation enforces this.
- All screenshot numbers are placeholders; no expected performance or counts may
  be inferred from them. Use the corrected researcher page with McNemar already present.

Implement each phase and satisfy its completion checks before moving to its
dependent phases. The plan defines the sequence; unresolved research choices
must be recorded explicitly rather than hidden in implementation defaults.

## Phase 1 - Foundation and contracts

**Purpose:** establish a runnable Python project and the rules its components
must share, before implementing model training.

Work:

- Inspect the available Python runtime and select a compatible dependency set.
  Configure `pyproject.toml`, a reproducible dependency lock, and the test runner.
  Keep OCR dependencies separate until the receipt approach is selected.
- Finalize the web framework choice; FastAPI is the current recommendation.
  Establish an application entry point and a health endpoint only.
- Implement application settings and path handling in `settings.py`. Keep local
  paths in environment settings and research settings in `experiment.yaml`.
- Define and validate the experiment configuration schema. The confirmed split
  is stratified 80/10/10. Unresolved parameters must be visibly unset and rejected
  when a dependent command is attempted; do not make illustrative numbers defaults.
- Define contracts for raw PaySim records, predictor inputs, separate ground-truth
  labels, paired model results, and experiment metadata. Labels are never features.
- Plan stable transaction identifiers so both models' outputs remain paired.
- Record methodology decisions in `docs/METHODOLOGY.md` and interface contracts
  in `docs/API.md` as those decisions are finalized. Mark open questions clearly.
- Document local installation, testing, and application startup commands.

Decisions to resolve before dependent model work:

- Exact dataset/version, final predictors, and timing conventions.
- Score definition, threshold and ties before training;
  PR-AUC numerical convention before Phase 4 evaluation.
- Scaling, SMOTE parameters, categorical-feature treatment, and fair RF tuning.
- SHAP output to explain, background selection, and explanation coverage.

Completion checks:

- A fresh local environment installs the package and imports `integritree`.
- Health and settings tests pass; configuration errors identify the missing or
  invalid field. No PaySim file or trained model is required for these checks.
- Paths work from the documented Windows commands without user-specific paths
  embedded in source code.
- The decision record separates settled choices from open research questions.

## Phase 2 - Dataset and preprocessing

**Requires:** Phase 1 contracts and the relevant feature/preprocessing decisions.

Work:

- Implement dataset loading and validation in `ml/data.py`. Preserve the raw CSV
  and record its fingerprint, row count, columns, label distribution, and audit.
- Handle duplicates, missing values, types, and invalid records using documented
  rules. Retain valid unusual transactions and report any removed records.
- Exclude the specified balance fields and `isFlaggedFraud` from predictors;
  keep `isFraud` separate. Define which original columns remain after engineering.
- Implement deterministic features in `ml/features.py` and fitted transformations
  in `ml/preprocessing.py`.
- Create and save shared stratified 80/10/10 split membership. Fit imputation and
  scaling on training records only and reuse the fitted objects for other splits.
- Implement `scripts/prepare_data.py` with saved audits, configuration, feature
  order, transformation metadata, and split provenance.

Completion checks:

- Splits are disjoint and reproducible; duplicate handling prevents identical
  original records from leaking across splits.
- Expected feature values match hand-worked synthetic examples. Changes to test
  values do not change fitted training medians or scaling parameters.
- Neither labels nor excluded columns reach the model feature matrix.
- Tiny datasets that cannot support the requested stratified split fail clearly.
- Validation/test data retain their original distributions; no SMOTE is applied.

## Phase 3 - Training and saved inference

**Requires:** Phase 2 and recorded RF/SMOTE/scoring decisions.

Work:

- Implement `ml/training.py` and `scripts/train_models.py` for RF and RF-SMOTE.
  Both start from the same original training records. Only the second branch
  applies the approved SMOTE procedure.
- Phase 3 implements the fixed baseline. The approved validation search follows
  Phase 4 metric implementation; see METHODOLOGY.md. Do not
  introduce class weighting or other differences that confound the comparison.
  If cross-validation is selected, fit preprocessing and SMOTE within each fold.
- Preserve the untouched test set during tuning. Record the final configuration
  and validation-based model selection before final test evaluation.
- Implement shared score/class prediction in `ml/inference.py` and artifact
  save/load in `ml/artifacts.py`.
- Save both models, fitted preprocessing, feature schema, threshold/score policy,
  dataset/split identifiers, seeds, package versions, and configuration together
  under `artifacts/<run_id>/`. Do not overwrite completed runs silently.

Completion checks:

- Tests demonstrate that SMOTE affects training only and preserves the pairing
  of evaluation records across both models.
- Reloaded artifacts reproduce the original predictions and scores within the
  defined numerical tolerance.
- Single-record and batch predictions agree for the same record, including
  feature order, class-label lookup, threshold behavior, and ties.
- Configuration/data combinations with too few fraud neighbors fail clearly.
- Validate the workflow on small synthetic fixtures before the full experiment.

Phase 3 verification at implementation: 123 tests passed, including matched RF settings, training-only
SMOTE, fractional indicators, reload parity, and individual/batch agreement.
Full PaySim baseline training is complete. Both models passed reload checks;
source/prepared hashes, training-row preprocessing replay, and individual/batch
prediction consistency passed full-run verification. See the methodology record.

## Phase 4 - Research evaluation and SHAP

**Requires:** working Phase 3 bundles and recorded evaluation/SHAP decisions.
Develop metrics on synthetic/validation data before tuning. Freeze settings and
the common cutoff before running official test evaluation.

Work:

- Add explicit evaluation-policy fields for undefined metrics, signed-comparison
  edge cases, McNemar handling, and method identifiers. Active YAML now selects
  signed_over_mean; the schema retains absolute_over_mean solely for historical
  compatibility. Read old snapshots unchanged. Create a separate evaluation
  configuration/report referencing the immutable trained bundle; do not overwrite
  its training configuration to enable evaluation or SHAP.
- Decide AP versus trapezoidal PR-AUC before reporting it. Keep the manuscript's
  continuity-corrected McNemar visible; settle a documented small-discordance rule
  before calculation. Zero discordances: p=1 by convention, statistic unavailable,
  status No discordant pairs. Do not silently substitute exact testing.

- Implement approved shared RF candidate/common-cutoff selection on validation F1,
  saving a selection record before official test evaluation.
  Candidates are (100,10), (100,20), (200,10), (200,20) for trees/depth. At 0.50,
  maximize mean validation F1 across both variants; ties favor shallower then fewer
  trees. For that shared configuration, search common cutoffs 0.05..0.95 by 0.05;
  ties favor nearest 0.50 then higher cutoff. Keep preprocessing and seeds fixed.
  Factor baseline-only validation out of the general artifact loader: it currently
  rejects non-0.50 thresholds and non-fixed_baseline runs. Introduce an explicit
  selected-run/selection-artifact contract and validate it without weakening
  legacy-bundle integrity. Preserve the original baseline, and test both loaders.
- Implement `ml/evaluation.py` and `scripts/evaluate_models.py` on the same held-out
  test records for both models. Full-PaySim training-record predictions must not
  be presented as the official test results.
- Compute precision, recall, F1, MCC, PR-AUC, and confusion matrices. Accuracy is
  supplementary. Use scores for PR-AUC and the agreed classes for hard metrics.
  Fraud is positive. Confusion matrices identify Actual and Predicted axes. MCC
  remains a -1..1 coefficient. Undefined metrics are null with a reason, not NaN,
  infinity, or an unexplained zero. Declare absent-class and undefined-F1 handling
  in the evaluation policy, including the validation-selection policy.
- Implement descriptive differences and the paired McNemar contingency table,
  statistic, p-value, and conclusion at alpha = 0.05. Describe differences in
  classification errors, not significance of each metric or automatic improvement.
- Resolve/report undefined metrics and zero-denominator percentage comparisons.
  Specify McNemar behavior for no or few discordant pairs, preserving the agreed
  statistical procedure rather than silently substituting a different test.
  Signed difference = 100*(S-B)/((S+B)/2), S=RF-SMOTE, B=benchmark RF. Use it for
  nonnegative values with positive mean. Both zero: null/zero_denominator. Either
  MCC negative: S-B with coefficient units and a different method label. Undefined
  inputs remain unavailable. Compute before rounding; positive means S is higher.
  Export the exact comparison method, units, and status. A descriptive difference
  never inherits statistical significance from the McNemar p-value.
- Implement `ml/explainability.py` for both models. Explain the same fraud output
  used by scoring and label the SHAP baseline, feature values, and contributions.
  Plan probability-space fraud-class explanations; finalize background strategy,
  size, seed, perturbation method, tolerance, and SHAP version before execution.
  A fixed sample of 200 original training rows is only a proposal, not a frozen
  setting. Do not use synthetic or held-out rows as background by convenience.
- Compute the top risk-increasing contributor as the largest contribution above
  the recorded positive tolerance, even for legitimate predictions. Break ties in
  saved feature order. Distinguish no positive contributor from pending/failed SHAP.
- Generate deterministic explanations from the same per-model contributions,
  using readable source values and both increasing/decreasing influences. Never
  cite excluded balance fields, invented behavior/history, or ground truth as an
  explanation. A zero type indicator means absence, not presence, of that type.
- Generate a model-specific waterfall asset with baseline, final output, readable
  feature values, signed contributions, units, and any grouped remaining effects.
  Supply a text description for accessibility. The expanded UI needs only this
  graph; keep numerical contributions internally for additivity and reproducibility.
- Define on-demand individual explanations and any reproducible sampling used
  for global summaries. Do not imply a sampled summary explains every record.
  Start with bounded page/detail explanation requests and cache by analysis/input
  revision, record, model run, and explainer version. Finalize measured limits;
  show status/coverage rather than pretending every uploaded record is explained.
  Global plots and full-dataset SHAP exports are optional future work.
- Export paired transaction predictions and scores, metrics, statistical results,
  and figures into `reports/<run_id>/` with matching artifact provenance.

Completion checks:

- Small hand-calculated examples verify confusion matrices, F1, MCC, percentage
  comparisons, and the paired contingency table.
- Test perfect predictions, all-legitimate predictions, absent classes, reversed
  model advantage, and no model disagreements; report unavailable results honestly.
- Verify that the SHAP baseline plus contributions reconstructs the explained
  output within tolerance, and that feature names match the saved feature order.
- Exported records retain identifiers and enough precision to reproduce metrics.
- RF-SMOTE is allowed to be better, worse, or statistically indistinguishable.
- Verify legacy and signed evaluation policies cannot be mislabeled, negative-MCC
  units and zero denominators, deterministic tie handling, and label/score alignment.
- Check legitimate/no-positive/negative-only explanations, per-model differences,
  chart/narrative agreement, and no reuse after input/model/explainer changes.
- Verify new selected-run loading preserves threshold ties and legacy inference;
  do not evaluate the official test set while developing or choosing settings.

## Phase 5 - Application services and API

**Requires:** verified Phase 3 inference and Phase 4 outputs.

Work:

- Implement services that reuse `ml/` functions for individual and batch inference,
  explanation requests, research report retrieval, and result exports.
- Implement `scripts/predict_batch.py` for compatible new records using saved
  models without refitting preprocessing or retraining.
- Implement prediction and research routes with the Phase 1 request/response
  contracts. Keep routes thin; do not duplicate model logic in API handlers.
  Extend the reserved contracts first: analysis/job identity, original/derived
  inputs, nullable ground truth, presentation bands, SHAP status/top contributor/
  narrative/chart, and downloads are not in the Phase 1 envelopes today.
- Distinguish labeled research inputs from unlabeled prediction inputs. Missing
  labels mean no evaluation metrics, correctness labels, or invented ground truth.
- Preserve the distinction between official held-out experiment results and
  additional uploaded labeled-dataset evaluations; labels alone prove neither
  compatibility nor independence from training data.
- Load trusted local model bundles, expose the model/run identifier, and return
  both models' results and disagreements explicitly.
  Keep run identity for provenance/cache/export. It does not mandate a visible
  model-run panel, correctness badge, or new disagreement filter.
- Add a downloadable CSV template/schema guide matching the ingestion contract.
  Phase 2's eleven-column, source-hash-bound research preparation must not become
  a general upload handler. Validate uploaded compatible predictor fields separately,
  preserve supplied original columns, and require independent isFraud for evaluation.
  Define the exact accepted upload schema before publishing the template. Never
  require excluded balance columns solely to construct the eleven predictors.
- Add server-side transaction-ID search and Model = Both/RF-SMOTE/Benchmark RF,
  Prediction outcome = All/TP/FP/TN/FN. Require one model and labels for an outcome;
  the UI disables/resets outcome to All for Both. Reject inconsistent API queries.
  Outcome filtering may compute correctness internally without displaying a badge.
  Search/filter precedes pagination over all records; keep stable source ordering,
  total/filtered counts, and return-navigation state. No score sort is required.
  Table filters do not silently redefine the official evaluation population.
- Return Original Inputs (as supplied/confirmed, with permitted traceability fields)
  and Derived Inputs (eleven engineered features, saved scaled values, feature order,
  transformations). Do not manufacture raw PaySim fields for receipts. Dataset
  Transaction ID means the original record identity, not a payment reference.
- Implement Download results as ZIP containing results.csv, evaluation.json,
  report.html, and metadata.json. The HTML report is readable offline, contains
  evaluation context/provenance and limitations, and escapes supplied text. CSV
  preserves identities/precision; protect spreadsheet readers from formula-like
  text. Include comparison methods/statuses and SHAP coverage. Raw SHAP CSV is
  optional and never implies full coverage. Missing data is not exported as zero.
  Distinguish all-results and filtered scopes; no giant HTML table or Excel truncation.
- Choose and implement a bounded local batch execution/progress approach based
  on measured workloads; large PaySim operations must not freeze the interface.
  Include validation previews and actionable errors; states for queued/processing,
  explaining, completed, partial failure, failed, and expired as applicable. Report
  measured progress or named indeterminate stages. Limit payload/page/export sizes,
  permit retry/replacement, and finalize lifecycle/cleanup. A database is optional;
  the temporary SQLite deduplication index in Phase 2 is not an application database.
  Associate access with the analysis, not a guessable transaction ID alone.
- Document invalid-input errors, missing-artifact behavior, file limits, and the
  frontend response contracts in `docs/API.md`.
  Update CORS methods/headers when actual upload/prediction routes exist; current
  GET-only settings are correct for health but not sufficient for those workflows.

Completion checks:

- API predictions match the scripts for identical records and artifacts.
- Research exports reproduce saved evaluation results.
- Unlabeled requests return predictions/explanations without evaluation results.
- Invalid schemas, missing models, incompatible features, and failed jobs return
  clear errors rather than mock or partial-success results.
- Batch progress, failure, and download behavior can be exercised locally.
- Test search/filter across page boundaries, invalid outcome/model combinations,
  source identity preservation, no-label responses, partial-SHAP exports, and
  exported/report values matching the declared evaluation population.
- Boundary-check 0,20,40,60,80,100 communication scores and rounding near boundaries;
  ensure band selection never changes the saved threshold classification.

## Phase 6 - GCash receipt demonstration

**Requires:** Phase 5 structured-record prediction and an agreed input mapping.

Work:

- Collect representative consented/redacted GCash person-to-person transfer
  layouts and select the OCR engine. Use synthetic receipt fixtures for Git.
- Implement `receipts/ocr.py` as an extraction adapter and `receipts/gcash.py` for
  supported fields such as amount, transaction date/time, and transfer type.
- Return editable extracted fields and missing/uncertain-field information.
  Prediction occurs only after user confirmation and required-field validation.
  Confirm receipt reference as string, amount/principal, date, unambiguous time,
  TRANSFER type, sender type and recipient type. Retain available masked names
  only for traceability; never reconstruct redacted identities or infer C/M from
  a personal name. Personal accounts are required for this supported workflow;
  unknown/merchant/contradictory roles block it. Separate internal analysis ID from
  receipt reference. Finalize missing-reference behavior; full names are optional.
- Advertise the five transaction categories but enforce TRANSFER-only receipt
  support on the server. Do not silently relabel a payment/cash-out as a transfer.
  All five categories remain supported by the research model/CSV workflow.
- Implement `receipts/mapping.py` using a documented mapping to the trained
  feature meanings. Resolve timing, amount units, category, and entity indicators
  explicitly; do not pretend PaySim steps are a verified GCash calendar.
- Add a distinct receipt/derived-input boundary, then share the saved scaling
  stage with raw PaySim preprocessing. Derive all eleven unscaled features and
  apply saved scaling once, preserving order. Do not make a fake step or C/M ID
  merely to pass PredictorInput. Do not accept arbitrary client-supplied scaled
  matrices as validated receipt inputs. Preserve raw-input behavior and state format.
  Missing original columns are acceptable only after this validated path exists.
- Freeze/version the demonstration mapping before enabling it: date/time convention,
  numeric PHP amount assumption, type/entity mapping, required fields and provenance.
  Neither derived features nor a disclaimer validates cross-domain equivalence.
  Store extracted and confirmed values with correction status and adapter version;
  a changed confirmation invalidates previous predictions/explanations.
- Where required model inputs cannot be supplied or mapped defensibly, return an
  explanation of the missing/unsupported input rather than fabricated values.
- Implement receipt routes and connect confirmed records to the existing
  prediction service. A manual form fallback remains a product decision to record.
- Decide temporary image retention/deletion and exclude personal receipt contents
  from logs and committed fixtures.
- Label results as experimental predictions of transaction behavior. Receipt
  forgery detection remains outside scope.
  Supply supported-scope and mapping information for Research Scope and Limitations
  and brief contextual receipt notices. Actual retention behavior must match the page.

Completion checks:

- Test supported layouts, blurry images, missing fields, invalid amount/date text,
  and unsupported transaction types or image formats.
- A corrected extraction reaches inference with the corrected values.
- Receipt and direct structured-record inputs produce identical results when the
  confirmed inputs match.
- Both model outputs are shown; no ground-truth metrics are fabricated.
- Test cleanup according to the agreed retention policy.
- Verify leading-zero references, masked/absent names, unsupported dropdown values
  submitted directly, ambiguous time, unknown roles, separate fees, and corrected data.
- Prove raw and derived paths produce identical scaled matrices/scores for equivalent
  synthetic PaySim inputs; this software parity is not real-GCash validity evidence.

## Phase 7 - Integration and reproducibility

**Requires:** Phases 1-6 completion checks.

Work:

- Validate backend contracts with both frontend views. Any frontend changes to
  replace mock data should be coordinated as an explicit integration task.
- Verify research report browsing/downloads and the full receipt-confirmation
  workflow. The unlabeled view must not display mock ground truth or metrics.
  Include waterfall-only details; missing/failing SHAP; both-model disagreement;
  no-positive contributors; original/derived input details; ground truth when known;
  disabled unsupported receipt types; table filters/pagination; and ZIP downloads.
- Align help modals in UI_HELP_CONTENT.md with backend method identifiers and
  formulas. Keep risk bands, predicted labels, signed differences and MCC units
  consistent across pages and exports. Verify accessible chart descriptions,
  keyboard dialogs, mobile/scroll behavior, and contextual limitations links.
- Run one documented, finalized experiment and preserve its artifacts, settings,
  provenance, reports, and interpretation. Use reproducibility checks to verify
  the recorded run, not to tune repeatedly against the test set.
- Record measured memory/runtime constraints and practical batch/SHAP limits.
  Computational optimization remains outside the thesis objectives.
- Finish installation/run instructions, the methodology decision record, API
  documentation, and the updated thesis context.

Completion checks:

- Another team member can follow the instructions to reproduce the workflow in
  the recorded environment using the same data and settings.
- Each displayed/exported research value is traceable to a run and saved outputs.
- Both user journeys work without hardcoded demonstration predictions.
- Document remaining limitations, particularly the lack of validation on real
  GCash transaction outcomes.

## Phase 1 verification

- Created an isolated Python 3.12.5 environment, installed hash-locked dependencies,
  and installed/imported the package in editable mode.
- All 70 foundation tests passed using synthetic inputs, including label/balance
  exclusion, paired model identities, configuration rejection, and command failures.
- A live local server started from the repository root returned HTTP 200 and the
  expected health JSON; the temporary server was stopped after verification.
- Dependency consistency check passed. The test client emits two upstream
  deprecation warnings (Starlette/httpx and the AnyIO portal alias); no test failed.
- The local dataset identity is recorded; Phase 1 did not prepare or train on it.

## Phase 2 verification

- All 106 tests passed, including 36 new preparation tests. Cases cover exact
  duplicates across batches, invalid/missing input, hand-worked features, tiny
  stratified datasets, no refitting on held-out values, saved-state replay,
  reproducible split membership, integrity checks, and failed-run rejection.
- Dependency locking/installation and `pip check` passed. The same two upstream
  test-client deprecation warnings recorded in Phase 1 remain.
- The full source run `paysim_phase2_20260918` retained 6,362,620 records:
  no exact duplicates, no missing fields, and no validation failures.
- Training: 5,090,096 rows / 6,570 fraud. Validation: 636,262 / 822 fraud.
  Testing: 636,262 / 821 fraud. Original class imbalance was preserved subject
  to integer rounding; no SMOTE was applied.
- A separate batched verification checked every feature row for finite values,
  binary indicators, unique source identity, and alignment with split/label
  membership. All source/bundle fingerprints matched. Saved preprocessing
  exactly reproduced 100 sampled records per split. Scratch files were cleaned.
- Local bundle: `backend/data/prepared/paysim_phase2_20260918/`.
  Verification report: `backend/reports/paysim_phase2_20260918/verification.json`.
  Both are generated/ignored outputs; installation and recreation commands are
  documented in the backend README. The raw CSV is unchanged.

## Next coding session: Phase 4

1. Finalize PR-AUC integration and small-discordance McNemar handling; translate
   approved undefined/comparison policies into explicit evaluation contracts.
2. Implement metrics, signed comparison and paired McNemar on synthetic fixtures.
   Separate evaluation configuration from immutable baseline artifact configuration.
3. Implement the approved bounded validation-selection workflow and compatible
   selected-run loading; freeze common settings/cutoff before official testing.
4. Finalize explainer settings/runtime coverage and implement top contributor,
   deterministic narrative, and waterfall-only detail output with additivity checks.
5. Publish evaluation/explanation report contracts for Phase 5. Keep receipt mapping,
   OCR, upload lifecycle/limits, and retention decisions with their dependent phases.

No full retraining is required solely for the reviewed UI labels, risk bands,
comparison formula, or waterfall presentation. The approved later validation search
is separate planned model work. Changing actual feature semantics would require a
new scientific decision and compatible new artifacts.

At each phase boundary, report implemented behavior, checks actually run,
remaining decisions, and changed files. Placeholder files and collected test
names are not evidence that a feature works.
