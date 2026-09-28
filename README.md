# Integritree

A Random Forest-Based Fraud Pattern Detection and Risk Scoring System for E-Wallet Transactions using SMOTE and SHAP Explainability.

Validation proceeds in three stages: **SMOTE ratio -> shared tree configuration ->
common threshold**. Current execution covers Stage 1 only, using the eight ratios
1:100, 1:50, 1:20, 1:10, 1:5, 1:3, 1:2, and 1:1. Results come from the fresh run; later stages remain pending.

Preparation, training, saved inference, evaluation, selection, and SHAP code are
implemented. Application prediction APIs, frontend integration, and the
experimental GCash receipt workflow remain planned. Expanded SHAP details use a
waterfall graph; a numerical table remains a suggestion.

- [Validation protocol](docs/THREE_STAGE_VALIDATION.md)
- [Current run status and reset record](docs/VALIDATION_RUN_STATUS.md)
- [Backend setup and commands](backend/README.md)
- [Methodology](docs/METHODOLOGY.md)
- [Agreed mock-up corrections](docs/MOCKUP_EVALUATION_DECISIONS.md)
- [Backend alignment audit](docs/BACKEND_ALIGNMENT_AUDIT.md)
- [Backend implementation plan](docs/BACKEND_IMPLEMENTATION_PLAN.md)
- [Help-modal and limitations-page content](docs/UI_HELP_CONTENT.md)
