# Integritree backend implementation plan

Last updated: 2026-09-18

## Current status and scope

The backend directory scaffold is complete. **Phase 1 code implementation has
not started.** Python files contain descriptive docstrings; configuration files
contain comments. They do not provide working commands, tests, or an API.

The initial backend supports two uses of the same saved RF and RF-SMOTE models:

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
| 1 | Foundation and contracts | Reproducible development setup and validated configuration/data contracts | Not started |
| 2 | Dataset and preprocessing | Audited, reproducible splits and training-fitted transformations | Not started |
| 3 | Training and saved inference | Both trained models and reloadable experiment bundles | Not started |
| 4 | Research evaluation and SHAP | Verified evaluation reports and model explanations | Not started |
| 5 | Application services and API | Research and individual structured-record workflows | Not started |
| 6 | GCash receipt demonstration | Image extraction, confirmation, mapping, and prediction | Not started |
| 7 | Integration and reproducibility | Backend handoff validated against the frontend workflows | Not started |

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
- Score definition, classification threshold, tie handling, and PR-AUC method.
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
- Use the shared validation set and a documented tuning budget/procedure. Do not
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

## Phase 4 - Research evaluation and SHAP

**Requires:** finalized Phase 3 bundles and recorded evaluation/SHAP decisions.

Work:

- Implement `ml/evaluation.py` and `scripts/evaluate_models.py` on the same held-out
  test records for both models. Full-PaySim training-record predictions must not
  be presented as the official test results.
- Compute precision, recall, F1, MCC, PR-AUC, and confusion matrices. Accuracy is
  supplementary. Use scores for PR-AUC and the agreed classes for hard metrics.
- Implement descriptive differences and the paired McNemar contingency table,
  statistic, p-value, and conclusion at alpha = 0.05. Describe differences in
  classification errors, not significance of each metric or automatic improvement.
- Resolve/report undefined metrics and zero-denominator percentage comparisons.
  Specify McNemar behavior for no or few discordant pairs, preserving the agreed
  statistical procedure rather than silently substituting a different test.
- Implement `ml/explainability.py` for both models. Explain the same fraud output
  used by scoring and label the SHAP baseline, feature values, and contributions.
- Define on-demand individual explanations and any reproducible sampling used
  for global summaries. Do not imply a sampled summary explains every record.
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

## Phase 5 - Application services and API

**Requires:** verified Phase 3 inference and Phase 4 outputs.

Work:

- Implement services that reuse `ml/` functions for individual and batch inference,
  explanation requests, research report retrieval, and result exports.
- Implement `scripts/predict_batch.py` for compatible new records using saved
  models without refitting preprocessing or retraining.
- Implement prediction and research routes with the Phase 1 request/response
  contracts. Keep routes thin; do not duplicate model logic in API handlers.
- Distinguish labeled research inputs from unlabeled prediction inputs. Missing
  labels mean no evaluation metrics, correctness labels, or invented ground truth.
- Preserve the distinction between official held-out experiment results and
  additional uploaded labeled-dataset evaluations; labels alone prove neither
  compatibility nor independence from training data.
- Load trusted local model bundles, expose the model/run identifier, and return
  both models' results and disagreements explicitly.
- Choose and implement a bounded local batch execution/progress approach based
  on measured workloads; large PaySim operations must not freeze the interface.
- Document invalid-input errors, missing-artifact behavior, file limits, and the
  frontend response contracts in `docs/API.md`.

Completion checks:

- API predictions match the scripts for identical records and artifacts.
- Research exports reproduce saved evaluation results.
- Unlabeled requests return predictions/explanations without evaluation results.
- Invalid schemas, missing models, incompatible features, and failed jobs return
  clear errors rather than mock or partial-success results.
- Batch progress, failure, and download behavior can be exercised locally.

## Phase 6 - GCash receipt demonstration

**Requires:** Phase 5 structured-record prediction and an agreed input mapping.

Work:

- Collect representative consented/redacted GCash person-to-person transfer
  layouts and select the OCR engine. Use synthetic receipt fixtures for Git.
- Implement `receipts/ocr.py` as an extraction adapter and `receipts/gcash.py` for
  supported fields such as amount, transaction date/time, and transfer type.
- Return editable extracted fields and missing/uncertain-field information.
  Prediction occurs only after user confirmation and required-field validation.
- Implement `receipts/mapping.py` using a documented mapping to the trained
  feature meanings. Resolve timing, amount units, category, and entity indicators
  explicitly; do not pretend PaySim steps are a verified GCash calendar.
- Where required model inputs cannot be supplied or mapped defensibly, return an
  explanation of the missing/unsupported input rather than fabricated values.
- Implement receipt routes and connect confirmed records to the existing
  prediction service. A manual form fallback remains a product decision to record.
- Decide temporary image retention/deletion and exclude personal receipt contents
  from logs and committed fixtures.
- Label results as experimental predictions of transaction behavior. Receipt
  forgery detection remains outside scope.

Completion checks:

- Test supported layouts, blurry images, missing fields, invalid amount/date text,
  and unsupported transaction types or image formats.
- A corrected extraction reaches inference with the corrected values.
- Receipt and direct structured-record inputs produce identical results when the
  confirmed inputs match.
- Both model outputs are shown; no ground-truth metrics are fabricated.
- Test cleanup according to the agreed retention policy.

## Phase 7 - Integration and reproducibility

**Requires:** Phases 1-6 completion checks.

Work:

- Validate backend contracts with both frontend views. Any frontend changes to
  replace mock data should be coordinated as an explicit integration task.
- Verify research report browsing/downloads and the full receipt-confirmation
  workflow. The unlabeled view must not display mock ground truth or metrics.
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

## Next coding session: Phase 1 only

Start with environment inspection, package/test setup, configuration validation,
shared contracts, and a health endpoint. Define settings before implementing
dataset processing. No model training, OCR, or final statistical conclusions are
part of Phase 1.

At each phase boundary, report implemented behavior, checks actually run,
remaining decisions, and changed files. Placeholder files and collected test
names are not evidence that a feature works.
