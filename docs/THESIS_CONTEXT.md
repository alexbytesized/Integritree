# Integritree — Thesis Context and Confirmed Requirements

Last reviewed: 2026-09-19

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

The researchers supplied `backend/data/raw/ps_raw.csv`: 6,362,620 data rows,
11 columns, and 8,213 fraud labels. Its SHA-256 and structural observations are
recorded in [METHODOLOGY.md](METHODOLOGY.md) and `backend/configs/experiment.yaml`.
The full Phase 2 audit found no exact duplicates or missing/invalid records.
All source rows were retained; preserve the raw CSV unchanged.

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
The researchers subsequently approved the Phase 2 preparation decisions below.
These implementation decisions have not been written into the manuscript.

### Approved Phase 2 preparation decisions

- Retain all 11 listed engineered features, including the constant/redundant
  merchant indicators. Exclude their original source columns from the model
  matrix while preserving source/audit records.
- Use `hour_of_day = (step - 1) % 24` and
  `day_of_week = ((step - 1) // 24) % 7`. These are simulated cycle positions,
  not verified local clock times or named weekdays.
- Remove exact duplicates across all 11 source columns before splitting, keeping
  the first occurrence and recording removals. Do not deduplicate feature vectors.
- Use training-only medians where meaningful and an explicit unknown-category
  representation. Never impute labels; stop/report malformed or unavailable
  time/entity inputs. Retain valid zero amounts and other unusual behavior.
- Min-Max scale log amount and both timing features using training data only;
  leave binary indicators unchanged and do not clip out-of-range transformed values.
- Keep stratified 80/10/10 splits, use split seed 42, and save shared membership.

The researchers accepted all six recommendations. Detailed policies and
implementation clarifications are in [METHODOLOGY.md](METHODOLOGY.md).
Phase 2 preparation is implemented and executed. The verified local bundle is
`backend/data/prepared/paysim_phase2_20260918/`. Training has 5,090,096 rows
(6,570 fraud), validation 636,262 (822 fraud), and testing 636,262 (821 fraud).
Preprocessing was fitted on original training records only. No SMOTE or RF
training has been performed. Full details are in the methodology decision record.

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
SMOTENC was discussed as a possible alternative. On 2026-09-19, the researchers
confirmed they cannot change to SMOTENC and will retain ordinary SMOTE. Its
fractional categorical-feature limitation must be documented and audited.

## 9. Decisions still unresolved

Research:

- SMOTE neighbor count and model/SMOTE seeds. Ordinary SMOTE and 1:1 balancing
  are approved; split seed 42 remains approved.
- Fixed versus tuned shared RF settings and a fair selection procedure.
  Matching RF settings are explicitly approved. Starting settings approved:
  100 trees, depth 20, split minimum 2, leaf minimum 1, sqrt feature selection,
  bootstrap enabled, no class weighting.
- Threshold search/tie-breaking and shared versus separate selected thresholds.
  Common baseline 0.50 with ties classified as fraud is approved, as is highest
  validation-set F1 for threshold selection. Final settings must precede test evaluation.
- PR-AUC calculation is now selected as Average Precision (AP). Label the
  precision-recall summary explicitly; AP is distinct from trapezoidal PR-AUC.
- Final frontend score display decisions. Mean-tree fraud probability from
  predict_proba is now approved as the internal 0..1 model risk score.
- SHAP background, output space, and computation coverage.

Receipt demonstration:

- Supported GCash receipt layouts and OCR engine.
- Mapping receipt fields to the trained feature meanings, including
  timing, amount units, transaction type, and merchant indicators.
- Handling incomplete, unreadable, and incompatible records.
- Manual-entry fallback.
- Upload retention and handling of personal receipt information.

Software:

- Dependency changes required for future SMOTE/SHAP/OCR work. Phases 1-2 use
  Python 3.12, FastAPI, NumPy, pandas, PyArrow, and scikit-learn with a versioned lock.
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
- Phase 1 is implemented: an installable Python 3.12 package, application settings,
  validated draft experiment configuration, shared data contracts, and a FastAPI
  health endpoint. Seventy foundation tests pass; the live health check and
  dependency consistency check also passed.
- Phase 2 is implemented: batched source validation, disk-backed exact duplicate
  checking, deterministic features, shared stratified splits, and saved
  training-fitted preprocessing. The full supplied dataset has been prepared.
- All 106 tests pass. Separate full-output checks verified every row for alignment
  and valid feature values, all file hashes, and preprocessing replay samples.
- `scripts/prepare_data.py` is functional. Training, evaluation, and batch
  prediction commands remain guarded placeholders.
- No training, prediction, evaluation, or OCR functionality is implemented.
  The frontend has not been connected to the backend.
- The raw PaySim file is available locally and remains unchanged.
- See [the backend implementation plan](BACKEND_IMPLEMENTATION_PLAN.md) and
  [backend setup instructions](../backend/README.md).

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
- The researchers authorized the backend scaffold, phased plan, and Phases 1-2
  implementation. Both phases are complete; Phases 3 onward remain pending.
  The six Phase 2 preparation recommendations were explicitly approved.
  Other open methodology choices are not approved by implementing contracts.
- Internal result contracts represent risk scores on a 0..1 scale; the score
  formula is now approved as mean-tree fraud probability. Any frontend scale
  choices and threshold optimization protocol still need to be specified.

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

## 13. Current Phase 3 decision status

Read the 2026-09-19 Phase 3 discussion in [METHODOLOGY.md](METHODOLOGY.md).
Approved choices have been written into the current experiment configuration;
the Phase 2 bundle retains its original configuration snapshot unchanged.

For ordinary SMOTE, keep generated fractional feature values as floating-point
training inputs. Do not round or truncate them into categorical indicators.
Original records keep valid indicators; synthetic labels remain fraud (1).
Audit synthetic feature validity and report the limitation. No resampling has run.

RF settings may be revised during development through new recorded training runs.
The user confirmed matching RF settings and selected highest validation-set F1
for threshold tuning from a common 0.50 baseline. Shared versus per-model final
thresholds, the search rules, and reporting protocol still need to be agreed.
AP is now selected as the precision-recall summary. Model/SMOTE seed 42 has
been explained but has not been explicitly approved. Phase 3 implementation has not
been authorized by this discussion alone.
