# Integritree

A Random Forest-Based Fraud Pattern Detection and Risk Scoring System for E-Wallet Transactions using SMOTE and SHAP Explainability.

Validation proceeds in three stages: **SMOTE ratio -> shared tree configuration ->
common threshold**. Stage 1 selected **1:100** from eight ratios; Stage 2 selected
**100 trees/depth 10/minimum leaf 1** from 12 configurations. Stage 3 selects a
shared cutoff from exactly **1%, 2%, ..., 100%** by mean validation F1.
See the current run status for measured results.

Preparation, training, saved inference, evaluation, selection, and SHAP code are
implemented, including the researcher CSV APIs and frontend integration. Individual
prediction APIs and the experimental five-category GCash receipt workflow remain
planned. The local receipt audit/OCR baseline and internal confirmed-input mapping
and paired prediction are implemented, with RapidOCR provisionally selected for integration. Expanded SHAP details use a waterfall graph; a numerical table remains
a suggestion.

- [Validation protocol](docs/THREE_STAGE_VALIDATION.md)
- [Current run status and reset record](docs/VALIDATION_RUN_STATUS.md)
- [Backend setup and commands](backend/README.md)
- [Methodology](docs/METHODOLOGY.md)
- [Agreed mock-up corrections](docs/MOCKUP_EVALUATION_DECISIONS.md)
- [Backend alignment audit](docs/BACKEND_ALIGNMENT_AUDIT.md)
- [Backend implementation plan](docs/BACKEND_IMPLEMENTATION_PLAN.md)
- [Approved receipt scope and screenshot collection guide](docs/RECEIPT_WORKFLOW.md)
- [Local OCR comparison, sample issues, and next implementation steps](docs/RECEIPT_OCR_BENCHMARK.md)
- [Implemented confirmed-receipt contracts and feature mapping](docs/RECEIPT_MAPPING.md)
- [Help-modal and limitations-page content](docs/UI_HELP_CONTENT.md)
