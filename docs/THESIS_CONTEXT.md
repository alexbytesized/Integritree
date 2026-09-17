# Integritree — Thesis Context and Confirmed Requirements

Last reviewed: 2026-09-18

## 1. Purpose of this document

This document preserves the thesis context and decisions confirmed by the
researchers. Read it before proposing changes or implementing features.

Distinguish:

- What the thesis manuscript specifies.
- What the researchers subsequently confirmed.
- Recommendations and unresolved decisions.

Do not treat recommendations as approved methodology changes.
Do not treat placeholder interface values as experimental findings.
Inspect the repository to establish the current implementation state.

## 2. Thesis identity and source

Tool name: Integritree

Thesis title:
“A Random Forest-Based Fraud Pattern Detection and Risk Scoring System
for E-Wallet Transactions Using SMOTE and SHAP Explainability”

Source manuscript:
`C:/Users/Carpicorn/Downloads/Thesis/COSC-305_CS-Thesis-Writing-1/Post-Proposal-Defense_Revision/Thesis-Proposal-Revised.docx`

Chapters reviewed:

1. The Problem and Its Setting
2. Review of Literature and Studies
3. Methodology

The researchers prefer to preserve Chapter 3 wherever possible.
Necessary corrections may be considered, but must be discussed.
The manuscript has not been edited in this conversation.

If the manuscript is unavailable in a future session, disclose that
limitation rather than claiming to have reread it.

## 3. Research objective

Develop and evaluate two models:

- RF: Random Forest trained on the original imbalanced training set.
- RF-SMOTE: Random Forest trained on a SMOTE-augmented version of that set.

The experiment investigates whether applying SMOTE changes fraud
detection performance.

Research questions:

1. What is the performance of RF?
2. What is the performance of RF-SMOTE?
3. Is there a statistically significant difference between the models?

SMOTE improving performance is a hypothesis to investigate, not an
outcome to assume. Mixed, negative, and nonsignificant findings remain
valid research results.

Random Forest performs classification.
SMOTE augments minority-class training examples.
The model output supplies the fraud risk score.
SHAP explains feature contributions to a model output.

## 4. Dataset and study boundaries

The study uses the synthetic PaySim mobile-money transaction dataset.
The ground-truth target is `isFraud`.

GCash and Maya motivate the Philippine context, but the study does not
train or evaluate using their actual transaction databases.

PaySim results do not establish real-world performance on Philippine
e-wallet transactions.

“Fraud pattern detection” means learning feature combinations associated
with the dataset’s fraud label. It does not mean identifying named scam
categories or verifying whether a transaction receipt is authentic.

Production deployment, scalability, and computational optimization are
outside the original research objectives.

## 5. Methodology specified in the manuscript

- Inspect missing values, duplicates, data types, and unusual values.
- Retain unusual values when they represent valid transaction behavior.
- Separate `isFraud` from predictor variables.
- Exclude:
  - `oldbalanceOrg`
  - `newbalanceOrig`
  - `oldbalanceDest`
  - `newbalanceDest`
  - `isFlaggedFraud`
- Use an 80% training, 10% validation, 10% testing stratified split.
- Preserve the original class distribution in validation and testing.
- Apply SMOTE only to the RF-SMOTE training branch.
- Use the same validation and test records for both models.
- Use validation for permitted model adjustments.
- Reserve the test set for final evaluation.

Features explicitly listed:

- Timing: `hour_of_day`, `day_of_week`.
- Transaction type: one-hot indicators for CASH_IN, CASH_OUT, DEBIT,
  PAYMENT, and TRANSFER.
- Amount: `log_amount = log(1 + amount)`, `is_zero_amount`.
- Entity type: `is_merchant_origin`, `is_merchant_dest`.

Do not silently add device information, transaction histories, or
frequency features mentioned only as general introductory examples.
The final treatment of original columns and precise feature mappings
still needs to be specified.

## 6. Evaluation and statistical interpretation

Primary metrics:

- Precision
- Recall
- F1-score
- Matthews Correlation Coefficient (MCC)
- Precision–Recall Area Under the Curve (PR-AUC)

Accuracy may be supplementary; it does not replace these five metrics.

Additional outputs:

- Confusion matrices.
- Descriptive percentage differences.
- McNemar contingency table, statistic, and p-value.
- Significance threshold: alpha = 0.05.

The researchers understand that McNemar’s test, applied to paired
correct/incorrect predictions, tests a difference in classification
error rates. It does not separately test significance for precision,
recall, F1, MCC, or PR-AUC.

A significant result can favor either model.
A nonsignificant result does not establish equivalence.

Ground-truth labels are required for evaluation and must never enter
the predictor inputs. Official thesis results must use held-out test
records, not the entire dataset after training.

PR-AUC requires prediction scores across thresholds, not just one
confusion matrix or hard predicted labels.

## 7. Confirmed application requirements

Preprocessing and model training run through separate Python scripts.

The application loads saved models and preprocessing components.
Prediction still applies preprocessing, but does not refit it on
uploaded records.

There are two views.

### Researcher view

Purpose: support the thesis experiment using compatible labeled data.

Display both models’ transaction predictions, risk scores, and SHAP
explanations, together with evaluation and statistical results.

Preserve transaction-level outputs suitable for checking calculations
and exporting research results.

The availability of a label column alone does not establish that an
arbitrary dataset is compatible with the trained model.

### Simple-user view

Purpose: an experimental demonstration of individual prediction.

Confirmed initial scope:

- GCash receipts.
- Person-to-person transfers.
- Predict possible fraud based on transaction behavior.
- Display both RF and RF-SMOTE results.
- No receipt forgery or image-manipulation detection.
- No evaluation metrics without independently established labels.

Intended flow:
Upload receipt → extract details → user confirms/completes fields →
validate and map inputs → preprocess → predict → explain.

Photo extraction supplies structured inputs; the Random Forest does
not classify the receipt’s pixels.

Show model disagreements honestly.
Do not invent missing fields or present predictions as verified facts.
PaySim evaluation does not validate real GCash receipt predictions.

## 8. Corrections discussed, not yet applied

The researchers acknowledged these corrections and will take note of
them. No blanket approval to rewrite Chapter 3 was given.

1. Learned preprocessing:
   Learn imputation and normalization parameters from training data
   only; reuse them for validation, testing, and individual inference.

2. Random Forest probability:
   The manuscript describes the proportion of trees voting fraud.
   Scikit-learn averages the trees’ class probability estimates.
   These can differ. Align the manuscript, implementation, prediction
   rule, and SHAP explanation target before finalizing the method.

3. Statistical conclusions:
   Describe McNemar’s scope precisely.
   Use “fail to reject the null hypothesis” for p >= 0.05.

4. Written equations:
   Correct F1 to 2TP / (2TP + FP + FN).
   Replace the apparent `TR` typo with `TP` in the MCC numerator.
   Absolute percentage difference reports magnitude, not direction.

A separate unresolved issue:
Ordinary SMOTE may produce fractional values in categorical indicators.
SMOTENC was discussed as a possible alternative, but changing to it
has NOT been approved.

## 9. Decisions still unresolved

Research:

- Exact PaySim file/version and final row counts.
- Final feature schema and retained original columns.
- Time-step origin and interpretation of derived day/hour features.
- Normalization method and categorical-feature treatment for SMOTE.
- SMOTE ratio, neighbor count, and random seeds.
- RF hyperparameters and a fair tuning procedure.
- Prediction threshold and tie handling.
- Exact PR-AUC calculation.
- Final risk-score definition and display scale.
- SHAP background, output space, and computation coverage.

Receipt demonstration:

- Supported GCash receipt layouts and OCR engine.
- Mapping receipt fields to the trained feature meanings, including
  timing, amount units, transaction type, and merchant indicators.
- Handling incomplete, unreadable, and incompatible records.
- Manual-entry fallback.
- Upload retention and handling of personal receipt information.

Software:

- Backend framework and dependency versions.
- Final API contracts and long-running batch handling.
- Whether persistence beyond local files is needed.

Do not silently resolve a methodological ambiguity merely to make
implementation easier.

## 10. Repository status at the time of review

- A React/Vite frontend exists.
- It includes upload, researcher-results, and user-results screens.
- Predictions, counts, and statistics currently include mock values.
- The user screen currently contains ground-truth information, which
  will need to reflect the agreed unlabeled-user workflow.
- The backend directory structure has been scaffolded with package modules,
  research scripts, configuration placeholders, test placeholders, and local
  data/artifact/report/runtime folders. The empty `test.txt` was removed.
- Python files contain descriptive docstrings only. Configuration files contain
  comments only; no dependencies have been installed for this scaffold.
- No training, prediction, evaluation, OCR, or API functionality is implemented.
- Phase 1 code implementation has not started. See
  [the backend implementation plan](BACKEND_IMPLEMENTATION_PLAN.md).

This is a dated observation. Reinspect before making changes.

## 11. Working constraints for future development

- Preserve the agreed RF-versus-RF-SMOTE experiment.
- Keep batch and individual prediction consistent.
- Keep training separate from application inference.
- Maintain alignment between each transaction and both models’ outputs.
- Preserve reproducible experiment settings and artifact provenance.
- Do not assume RF-SMOTE must outperform RF.
- Do not confuse experimental receipt prediction with validated
  real-world fraud detection.
- The researchers authorized the backend directory scaffold and a phased
  implementation plan. The structure now exists, but functional code is deferred
  to later phases. Framework, methodology, and open parameter choices are not
  approved merely by creating these folders.

## 12. Technical references for the discussed corrections

- Preprocessing and leakage:
  https://scikit-learn.org/stable/common_pitfalls.html
- Random Forest prediction/probability behavior:
  https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
- F1 calculation:
  https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
- McNemar implementation:
  https://www.statsmodels.org/stable/generated/statsmodels.stats.contingency_tables.mcnemar.html
- SMOTE and categorical data:
  https://imbalanced-learn.org/stable/over_sampling.html
