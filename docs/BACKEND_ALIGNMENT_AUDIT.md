# Backend alignment audit against the agreed mock-up review

Historical alignment audit, updated for the three-stage cleanup on 2026-09-28.
The researcher application integration described as pending below was implemented
on 2026-09-29; see [current workflow and verification](RESEARCHER_WORKFLOW.md).
Receipt integration and final evaluation remain pending.
Receipt requirements below were updated on 2026-10-01 for the approved
[five-category scope](RECEIPT_WORKFLOW.md), superseding the earlier TRANSFER-only
restriction. This updates requirements, not the historical code-review findings.
The [local OCR baseline](RECEIPT_OCR_BENCHMARK.md) and
[confirmed-receipt mapping/paired prediction](RECEIPT_MAPPING.md) are now implemented;
application receipt routes remain pending. The historical code-review table below
describes the earlier checkpoint, not the current receipt module status.

## Result and scope

The reusable foundation supports PaySim preparation, paired training/inference,
evaluation, three-stage selection, and SHAP. Application integration remains
pending. See [the protocol](THREE_STAGE_VALIDATION.md) and
[current run status](VALIDATION_RUN_STATUS.md) for research execution evidence.
The audit addresses alignment with [the mock-up decisions](MOCKUP_EVALUATION_DECISIONS.md),
not model performance. Expanded SHAP details use a waterfall; a numerical table
remains suggestion-only.

## Completed-code review

| Area | Code inspected | Finding and disposition |
| --- | --- | --- |
| Foundation / availability | config.py, contracts.py, api/schemas.py, api/main.py | Strict finite typed contracts and health-only API match the implemented scope. Existing envelopes are reserved foundations; extend them for application results in Phase 5. |
| Comparison setting | config.py, configs/experiment.yaml | Previously accepted only absolute_over_mean. Active YAML now selects signed_over_mean; schema accepts both and retains the historical default when an older snapshot omits the field. The formula and explicit exported policy are implemented. |
| Dataset parsing and identity | ml/data.py, ml/preparation.py, contracts.py | Raw research preparation validates the original eleven columns and approved source fingerprint, deduplicates typed original records, and preserves original source row identity. The record ID is dataset hash plus row, not a payment reference. This is not a general upload handler. |
| Features / exclusions | ml/features.py | Exactly eleven agreed predictors derived from five raw fields; balances, flags, names as categories, and labels are absent from the feature matrix. Simulation day/hour formulas retain their agreed interpretation. |
| Training-fitted transformations | ml/preprocessing.py, ml/preparation.py | Amount imputation and scaling fit only original training records; binary features remain unscaled; transform reuses saved state without clipping/refitting. Original data and split membership remain available for later record details. |
| Paired training | ml/training.py | Matched RF settings, ordinary SMOTE only in the second training branch at the candidate ratio, floating-point synthetic features preserved. Trainer loads only the training split. The three-stage protocol selects ratio, forest, and common threshold. |
| Scores / class decisions | ml/inference.py | Uses fraud-class predict_proba with class lookup, finite 0..1 scores, shared >= threshold tie behavior, and aligned IDs for both models. Display bands and Predicted labels are future presentation mapping, not changes to these probabilities. |
| Saved models | ml/artifacts.py, ml/training.py | Manifest hashes, package versions, feature order, classes and matched settings are checked. Ordinary fitted bundles and completed schema-2 three-stage selections have separate integrity checks. Partial and obsolete selections are rejected. |
| Derived-only / receipt inputs | contracts.py, ml/preprocessing.py, ml/inference.py | Current predict_records requires step/type/amount/nameOrig/nameDest. Low-level predict_features consumes an already prepared matrix; it is not a validated receipt adapter. Phase 6 needs separate confirmed-receipt validation and shared scaling, with no fabricated step or entity IDs. |
| SHAP data contract | contracts.py, ml/explainability.py | Numerical contributions, computation, reconstruction checks, and waterfall output are implemented. Keep internal contributions for graph/narrative/additivity. Their existence does not require an on-screen numerical table. Integrate the implemented status, identity, narrative, top-positive contributor, and chart output into application services. |
| Evaluation and application modules | ml/evaluation.py, services/, api/routes/, receipts/, later-phase scripts/tests | Evaluation is implemented and tested. Services, prediction routes, and receipts remain placeholders. Mock screen values are not evidence of completed evaluation. |

Paths in the table are relative to backend/src/integritree unless a configuration
or tests/scripts path is stated. The audit inspected implementation and synthetic
tests; it is not an exhaustive security or scientific-validity certification.

## Required additions, with ownership

| Requirement | Phase | Completion evidence |
| --- | --- | --- |
| Five metrics, confusion matrices, signed comparison, paired McNemar | 4 | Hand-calculated fixtures, undefined-value policies, selected methods and precise units in reports. |
| Validation selection and immutable final-run provenance | 4 | Three-stage procedure, ordinary and selected-run reload tests, rejection of obsolete selection bundles, test set unused for selection. |
| Top risk-increasing contributor, deterministic narrative, waterfall | 4 | Per-model additivity, signs, legitimate/no-positive/failure cases, graph/narrative agreement. Numerical table not required. |
| Research uploads and downloadable template | 5 | Actual accepted schema, label separation, validation preview/errors, source identity preservation. Do not rerun preparation/training for uploads. |
| Original Inputs / Derived Inputs | 5-6 | Original/confirmed values plus engineered/scaled values and provenance; unavailable original fields remain unavailable. |
| Risk bands and labels | 5 | Unrounded boundary tests; score bands do not override the saved classification threshold. |
| Table search, model/outcome filters, pagination | 5 | Whole-dataset querying before paging, one-model/ground-truth requirements for TP/FP/TN/FN, unchanged evaluation population. |
| ZIP Download results | 5 | results.csv, evaluation.json, offline report.html, metadata.json; exact methods, scope, coverage, precision and missing-value statuses. |
| Five-category GCash app receipt demonstration | 6 | Per-workflow extraction, correction/confirmation, supported-layout/category/account-role validation, approved time/amount assumptions and shared-scaler parity. RapidOCR is provisionally selected; confirmed-role enforcement/shared-scaler parity are implemented internally; layout release checks, application integration and cash-agent mappings remain pending. |
| Limitations/help and full user journeys | 7 | Method-aware help, contextual notices, waterfall accessibility, unlabeled receipt results, failure/recovery and export checks. |

No new on-screen correctness badge, model-run panel, or evaluation metadata panel
is required. Retain those facts internally or in exports as appropriate. Keep the
approved color-only visible score/contributor design and no risk-score sorting.
No persistent application database is mandated. Phase 2's temporary SQLite index
supports deduplication only and is cleaned up by preparation.

## Verification and remaining work

Synthetic tests cover configuration, preparation, paired training and inference,
metrics, three-stage freezing/resume/integrity, and SHAP equivalence. See
[current verification](VALIDATION_RUN_STATUS.md) for the full suite result and
retained-data preservation checks.

PR-AUC, McNemar, and SHAP methods are settled in the
[methodology](METHODOLOGY.md). Receipt layout release checks, cash-agent mappings,
job/API contracts and exact ZIP files remain to be finalized. RapidOCR is provisionally
selected. Personal-wallet mappings for TRANSFER/PAYMENT/DEBIT, one PNG/JPEG up to
10 MiB / 20 million pixels, staged release, post-upload completion, optional reference,
time/amount assumptions, ZIP contents without images/metrics, and temporary storage
until Clear or backend shutdown/restart are settled requirements. See
[receipt decisions](RECEIPT_WORKFLOW.md) for status and implementation sequence.
Mock-up changes do not require altering the retained raw/prepared data.
