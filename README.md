# Integritree

A Random Forest-Based Fraud Pattern Detection and Risk Scoring System for E-Wallet Transactions using SMOTE and SHAP Explainability.

Validation proceeds in three stages: **SMOTE ratio -> shared tree configuration ->
common threshold**. Stage 1 selected **1:100** from eight ratios. Current execution
compares the 12 shared forest configurations in Stage 2 and stops before threshold
selection. Graceful pausing saves the current paired candidate before exiting.

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
