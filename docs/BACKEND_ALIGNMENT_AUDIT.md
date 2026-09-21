# Backend alignment audit against the agreed mock-up review

Reviewed: 2026-09-22

## Result and scope

Phases 1-3 align with the approved PaySim preparation and baseline-model method.
They are a reusable foundation, not an implementation of all reviewed screens.
The current audit found one configuration mismatch and several future integration
dependencies. The mismatch is corrected: active configuration now selects the
approved signed symmetric percentage difference and still reads legacy snapshots.
No metrics, SHAP, receipt workflow, or application prediction API was implemented
by this audit. Full training and official test evaluation were not rerun.

The mock-up can be revised using [the decision record](MOCKUP_EVALUATION_DECISIONS.md).
The expanded SHAP view is now waterfall-only. Its numerical table remains a
suggestion. [The updated implementation plan](BACKEND_IMPLEMENTATION_PLAN.md)
assigns the remaining work and checks to Phases 4-7.

## Completed-code review

| Area | Code inspected | Finding and disposition |
| --- | --- | --- |
| Foundation / availability | config.py, contracts.py, api/schemas.py, api/main.py | Strict finite typed contracts and health-only API match the implemented scope. Existing envelopes are reserved foundations; extend them for application results in Phase 5. |
| Comparison setting | config.py, configs/experiment.yaml | Previously accepted only absolute_over_mean. Active YAML now selects signed_over_mean; schema accepts both and retains the historical default when an older snapshot omits the field. Phase 4 must implement the actual formula and export method. |
| Dataset parsing and identity | ml/data.py, ml/preparation.py, contracts.py | Raw research preparation validates the original eleven columns and approved source fingerprint, deduplicates typed original records, and preserves original source row identity. The record ID is dataset hash plus row, not a payment reference. This is not a general upload handler. |
| Features / exclusions | ml/features.py | Exactly eleven agreed predictors derived from five raw fields; balances, flags, names as categories, and labels are absent from the feature matrix. Simulation day/hour formulas retain their agreed interpretation. |
| Training-fitted transformations | ml/preprocessing.py, ml/preparation.py | Amount imputation and scaling fit only original training records; binary features remain unscaled; transform reuses saved state without clipping/refitting. Original data and split membership remain available for later record details. |
| Paired training | ml/training.py | Matched RF settings, ordinary SMOTE only in the second training branch, 1:1 balance, floating-point synthetic features preserved. Trainer loads only the training split. No new mock-up requirement requires changing this baseline experiment. |
| Scores / class decisions | ml/inference.py | Uses fraud-class predict_proba with class lookup, finite 0..1 scores, shared >= threshold tie behavior, and aligned IDs for both models. Display bands and Predicted labels are future presentation mapping, not changes to these probabilities. |
| Saved models | ml/artifacts.py, ml/training.py | Manifest hashes, package versions, feature order, classes and matched settings are checked. Loader deliberately calls baseline validation: a tuned non-0.50 threshold cannot be loaded as a current Phase 3 bundle. Phase 4 must add an explicit selected-run contract while preserving legacy validation. |
| Derived-only / receipt inputs | contracts.py, ml/preprocessing.py, ml/inference.py | Current predict_records requires step/type/amount/nameOrig/nameDest. Low-level predict_features consumes an already prepared matrix; it is not a validated receipt adapter. Phase 6 needs separate confirmed-receipt validation and shared scaling, with no fabricated step or entity IDs. |
| SHAP data contract | contracts.py, ml/explainability.py | A numerical contribution contract exists; computation does not. Keep internal contributions for graph/narrative/additivity. Their existence does not require an on-screen numerical table. Add statuses, target/model identity, narrative, top-positive selection, and chart output in Phases 4-5. |
| Evaluation and application modules | ml/evaluation.py, services/, api/routes/, receipts/, later-phase scripts/tests | Placeholders remain clearly marked. No metric/receipt tests are executable in the placeholder test files. Model fields and mock screen values are not evidence of completed evaluation. |

Paths in the table are relative to backend/src/integritree unless a configuration
or tests/scripts path is stated. The audit inspected implementation and synthetic
tests; it is not an exhaustive security or scientific-validity certification.

## Required additions, with ownership

| Requirement | Phase | Completion evidence |
| --- | --- | --- |
| Five metrics, confusion matrices, signed comparison, paired McNemar | 4 | Hand-calculated fixtures, undefined-value policies, selected methods and precise units in reports. |
| Validation selection and immutable final-run provenance | 4 | Approved candidate/cutoff procedure, legacy and selected-run reload tests, test set unused for selection. |
| Top risk-increasing contributor, deterministic narrative, waterfall | 4 | Per-model additivity, signs, legitimate/no-positive/failure cases, graph/narrative agreement. Numerical table not required. |
| Research uploads and downloadable template | 5 | Actual accepted schema, label separation, validation preview/errors, source identity preservation. Do not rerun preparation/training for uploads. |
| Original Inputs / Derived Inputs | 5-6 | Original/confirmed values plus engineered/scaled values and provenance; unavailable original fields remain unavailable. |
| Risk bands and labels | 5 | Unrounded boundary tests; score bands do not override the saved classification threshold. |
| Table search, model/outcome filters, pagination | 5 | Whole-dataset querying before paging, one-model/ground-truth requirements for TP/FP/TN/FN, unchanged evaluation population. |
| ZIP Download results | 5 | results.csv, evaluation.json, offline report.html, metadata.json; exact methods, scope, coverage, precision and missing-value statuses. |
| GCash person-to-person receipt demonstration | 6 | Extraction, correction, confirmation, TRANSFER-only server enforcement, explicit time/amount assumptions and shared-scaler parity. |
| Limitations/help and full user journeys | 7 | Method-aware help, contextual notices, waterfall accessibility, unlabeled receipt results, failure/recovery and export checks. |

No new on-screen correctness badge, model-run panel, or evaluation metadata panel
is required. Retain those facts internally or in exports as appropriate. Keep the
approved color-only visible score/contributor design and no risk-score sorting.
No persistent application database is mandated. Phase 2's temporary SQLite index
supports deduplication only and is cleaned up by preparation.

## Checks actually performed

- Inspected source/configuration, existing tests, and Phase 2/3 verification reports.
- Ran `.\backend\.venv\Scripts\python.exe -m pytest backend/tests -q` from the
  repository root: **125 passed**, two existing upstream test-client deprecation
  warnings, 47.40 seconds.
- Added regression coverage for legacy/default comparison-policy loading, invalid
  comparison names, and synthetic training/reload/inference under both signed and
  legacy configurations. The synthetic preparation-to-training check also covers
  an evaluation-policy-only change without invalidating feature preparation.
- Ran configuration readiness for train: valid, no unresolved train fields.
- Ran evaluation readiness: correctly rejected unresolved pr_auc_method,
  mcnemar_method, and discordance_edge_policy. Accepting the signed policy does
  not falsely make evaluation ready.
- Read the prior full-run reports: both report passed integrity/preprocessing
  checks, and the Phase 3 report explicitly states no test/validation evaluation.
  Those expensive full-dataset checks were not repeated during this audit.
- Checked the actual saved Phase 2 and Phase 3 configuration fingerprints against
  their manifests and parsed both with the updated schema: both retain their
  original absolute_over_mean policy unchanged. This was a configuration check,
  not a fresh full-model reload or full-dataset verification.
- Checked local Markdown link targets across the project documentation and ran
  whitespace checks on the edited tracked files; no broken local targets or
  whitespace errors were found.

## Remaining choices and limits

UI correction decisions are established; some dependent backend details still need
finalization: PR-AUC integration, small-discordance McNemar handling, SHAP settings
and runtime coverage, receipt mapping/layout/OCR, missing-reference/manual-fallback
behavior, and upload/export retention and limits. Keep these explicit in their
own phase rather than reopening settled UI decisions or inventing scientific defaults.

The backend changes in this audit are limited to comparison configuration/schema
compatibility and regression tests. Raw/prepared data, saved trained artifacts,
model settings, inference code, frontend components, and thesis manuscript were
not changed. No retraining is needed solely for the UI decisions reviewed here.
