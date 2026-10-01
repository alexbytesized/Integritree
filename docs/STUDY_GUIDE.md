# Integritree study guide

Reviewed against the local code and recorded evidence on **2026-10-01**.

Integritree is a research prototype that compares Random Forest (RF) with Random
Forest trained using SMOTE (RF-SMOTE). It uses synthetic PaySim transactions for
the research experiment, provides a labeled-CSV evaluation interface, and accepts
supported GCash screenshots for an experimental receipt demonstration.

## Reading order

| Guide | What you will learn | Suggested first reading |
| --- | --- | --- |
| [Terms and concepts](STUDY_TERMS.md) | Data, models, scores, metrics, SHAP, receipts, and software vocabulary | Read the worked metric example and common misunderstandings. |
| [Code and system guide](STUDY_CODE_GUIDE.md) | Tech stack, folders, important functions, request flow, and where to make changes | Follow one CSV and one receipt through the system. |
| [Methodology and remaining work](STUDY_METHODOLOGY_GUIDE.md) | What the experiment did, why the steps matter, measured status, and what remains | Read the status table, selection stages, and final-evaluation checklist. |

You do not need to memorize every function. Start by explaining the path from an
input to its features, two model scores, predicted classes, and explanations.
Then study where evaluation labels enter and why they never become predictors.

## The current system in one page

| Topic | Current position |
| --- | --- |
| Research dataset | 6,362,620 synthetic PaySim records; 8,213 labeled fraud. |
| Preparation | Completed; stratified 80/10/10 train/validation/test split. |
| Predictors | Eleven engineered features; balances and `isFlaggedFraud` excluded. |
| Compared models | RF and RF-SMOTE; both are retained. |
| Selected settings | SMOTE fraud:legitimate ratio 1:100; 100 trees; depth 10; minimum leaf size 1. |
| Decision rule | Fraud when the unrounded score is at least **0.43**, for either model. |
| Selection status | All three validation stages completed; Stage 3 finished September 29, 2026. |
| Final thesis testing | Official held-out test evaluation remains pending. |
| Researcher interface | Labeled CSV upload, paired predictions, evaluation, requested SHAP, and ZIP export. |
| Receipt interface | Local OCR, confirmation, mapping v2, paired predictions/SHAP, ZIP export, and cleanup. |
| Recognized receipt layouts | Express Send, Pay Online, and Bank Transfer Complete. Cash In/Out and merchant QR recognition remain pending. |

Receipt extraction, model prediction, and model evaluation answer different
questions. Reading an amount correctly does not establish that a model detects
real GCash fraud. A model prediction does not verify that a receipt is authentic.

## Which records to trust for which question

These guides explain the current implementation; they do not replace the frozen
experiment records or create new research decisions.

- **Exact selection rules:** [THREE_STAGE_VALIDATION.md](THREE_STAGE_VALIDATION.md)
  and [validation configuration](../backend/configs/validation_three_stage.yaml).
- **What actually finished:** [VALIDATION_RUN_STATUS.md](VALIDATION_RUN_STATUS.md)
  and its linked local reports.
- **Researcher operation:** [RESEARCHER_WORKFLOW.md](RESEARCHER_WORKFLOW.md).
- **Receipt behavior and current role mapping:**
  [RECEIPT_APPLICATION.md](RECEIPT_APPLICATION.md) and
  [RECEIPT_MAPPING.md](RECEIPT_MAPPING.md).
- **OCR evidence:** [RECEIPT_OCR_BENCHMARK.md](RECEIPT_OCR_BENCHMARK.md).
- **Changes still to apply to the thesis:**
  [THESIS_DOCUMENT_CHANGES.md](THESIS_DOCUMENT_CHANGES.md).

Some older phase narratives in `METHODOLOGY.md`, planning files, and protocol
execution examples retain earlier pending-work language. Read them as historical
context where they conflict with the current run status or connected application.
The generic `experiment.yaml` defaults are also different from the finalized
selected bundle; this is explained in the code guide.

## Suggested study session

1. Explain the project in two sentences without claiming that SMOTE must win.
2. Trace a CSV record through validation, feature engineering, saved scaling,
   both models, and evaluation.
3. Trace a receipt through OCR, user confirmation, and the same saved models.
4. Explain why a 45% score can be classified as fraud while its display band is
   Moderate Risk.
5. Explain the difference between validation results, software tests, OCR checks,
   and the pending final thesis evaluation.
6. Use the questions at the end of each guide to rehearse as a group.

Documentation preparation did not run training, selection, or held-out evaluation.
