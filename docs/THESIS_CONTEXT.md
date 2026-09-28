# Integritree - Thesis Context and Confirmed Requirements

The sole selection workflow is [three-stage validation](THREE_STAGE_VALIDATION.md):
SMOTE ratio by RF-SMOTE validation Average Precision, shared forest settings by
mean validation AP, then a common threshold by exact mean validation F1. Each
stage freezes before the next. Both models are retained; selection does not pick
a model winner. Stage 1 froze 1:100; current execution stops after Stage 2. See
[run status](VALIDATION_RUN_STATUS.md) for measured outcomes.

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
Preprocessing was fitted on original training records only. It is retained for
the fresh validation run; see the current run status for training progress.

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
   The manuscript's absolute percentage difference reports magnitude, not direction.
   The researchers subsequently approved signed symmetric percentage difference;
   revise Equation 11 and its interpretation using METHODOLOGY.md.

An acknowledged limitation of the approved method:
Ordinary SMOTE may produce fractional values in categorical indicators.
SMOTENC was discussed as a possible alternative. On 2026-09-19, the researchers
confirmed they cannot change to SMOTENC and will retain ordinary SMOTE. Its
fractional categorical-feature limitation must be documented and audited.

## 9. Decisions still unresolved

Research:

- PR-AUC convention is settled: Average Precision (AP), approved 2026-09-23.
  Implemented in Phase 4 configuration/calculation; clarify manuscript Equation 10.
- McNemar is settled and implemented: continuity-corrected primary; exact two-sided
  binomial supplement for 1-24 discordances; zero discordances give p=1 by convention
  and no statistic. Signed comparison and undefined-value policies are implemented.
- Final RF configuration and cutoff selected by the approved future validation
  search; its procedure is settled, its winning settings are not known.
- SHAP is settled and implemented: interventional fraud-probability TreeSHAP,
  shared uniform 200-record original-training background, seed 42. Coverage is
  on demand, a shared uniform 1,000-record global sample (seed 42), or an explicit
  full-report job. Reconstruction tolerance is 1e-6; positive tolerance is 1e-9.
- Application job scheduling/lifecycle remain open; the score display is now 0..100 with
  approved communication bands, while internal scores remain 0..1.

Ordinary SMOTE uses k_neighbors=5 and seed 42; both forests use model seed 42
and matching settings. The three-stage protocol determines ratio, forest, and
common cutoff. No per-model thresholds are planned.

Receipt demonstration:

- Supported GCash receipt layouts and OCR engine.
- Mapping receipt fields to the trained feature meanings, including
  timing, amount units, transaction type, and merchant indicators.
- Handling incomplete, unreadable, and incompatible records.
- Manual-entry fallback.
- Upload retention and handling of personal receipt information.

Software:

- Dependency changes required for future OCR work. Phases 1-4 use
  Python 3.12, FastAPI, NumPy, pandas, PyArrow, scikit-learn, imbalanced-learn,
  joblib, SciPy, Matplotlib, and SHAP 0.52.0 with a versioned lock.
- Final API contracts and long-running batch handling.
- Whether persistence beyond local files is needed.

Do not silently resolve a methodological ambiguity merely to make
implementation easier.

## 10. Repository status

The React/Vite frontend includes upload and results screens with mock values;
these are not experimental results. Python implements preparation, training,
saved inference, evaluation, three-stage selection, and SHAP. HTTP is health-only;
prediction routes, frontend integration, general batch exports, and OCR remain
future work. The raw dataset, completed preparation, and preparation verification
are retained. Current model/validation execution is recorded in
[run status](VALIDATION_RUN_STATUS.md).

## 11. Working constraints for future development

- Preserve the agreed RF-versus-RF-SMOTE experiment.
- Keep batch and individual prediction consistent.
- Keep training separate from application inference.
- Maintain alignment between each transaction and both models’ outputs.
- Preserve reproducible experiment settings and artifact provenance.
- Do not assume RF-SMOTE must outperform RF.
- Do not confuse experimental receipt prediction with validated
  real-world fraud detection.
- The researchers authorized the scaffold, phased plan, and Phases 1-3 code.
  Training code is implemented; current run status determines available models.
  The six Phase 2 preparation recommendations were explicitly approved.
  Other open methodology choices are not approved by implementing contracts.
- Internal result contracts represent risk scores on a 0..1 scale; the score
  formula is mean-tree fraud probability. Frontend scale is 0..100 with five
  communication bands; the separate shared threshold search protocol is approved.

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

## 13. Training and execution boundaries

The three-stage protocol is the only supported selection workflow. Generic
fixed-parameter training is a reusable component, not a separate tuning method.
Stage 1 is complete and its 1:100 ratio is frozen. The continuation compares
all 12 forests and stops after Stage 2. Threshold selection, official testing,
and SHAP execution remain outside this run. The Word reference documents have not been edited; use
[the manuscript checklist](THESIS_DOCUMENT_CHANGES.md) for alignment.

## 14. Completed mock-up review and backend alignment (2026-09-22)

Use [MOCKUP_EVALUATION_DECISIONS.md](MOCKUP_EVALUATION_DECISIONS.md) for the agreed
screen corrections and [UI_HELP_CONTENT.md](UI_HELP_CONTENT.md) for standardized
help and Research Scope and Limitations content. Use the corrected researcher
screen with McNemar, and ignore all screenshot sample numbers.

- SHAP: top risk-increasing contributor, deterministic explanation, waterfall-only
  expanded details. Numerical table remains a suggestion, outside implementation.
- Original Inputs / Derived Inputs and known ground truth are shown in details;
  no extra correctness badge or model-run panel. No on-screen evaluation metadata
  panel; relevant provenance and evaluation context go in the approved ZIP download.
- Retain color-only visible model association, horizontal table scrolling and no
  score sorting. Model and Prediction outcome filters follow the decision record.
- Predicted Fraud / Predicted Legitimate; score bands [0,20), [20,40), [40,60),
  [60,80), [80,100] are Minimal through Critical, independent of class threshold.
- Signed symmetric percentage difference is approved. Active YAML now selects
  signed_over_mean; historical absolute configurations remain readable/unchanged.
  Metric calculation and edge-case policies are implemented in Phase 4.
- GCash person-to-person receipt workflow only; TRANSFER selectable, other types
  visible but disabled, server-enforced. Exact names remain optional traceability
  information and never predictors. Receipt timing/amount mapping is experimental.

[The backend audit](BACKEND_ALIGNMENT_AUDIT.md) found the completed preparation and
baseline method aligned and documented future extensions: selected-run artifacts,
evaluation/SHAP, application results/query/export contracts, and a receipt adapter
using shared saved scaling. Evaluation, three-stage selection, and SHAP code are now implemented. No new model
training or official performance result was produced by this review.


## 15. Evaluation and explanation implementation

Continuity-corrected McNemar with its small-discordance supplement and shared
training-reference TreeSHAP are implemented. Read
[implementation details](PHASE4_IMPLEMENTATION.md) for selected-bundle guards,
metric policies, coverage, and the prediction-equivalent SHAP adapter.
Only the [current run record](VALIDATION_RUN_STATUS.md) establishes which research
stages have completed. A ratio decision does not finalize a forest or threshold.
The next software phase is application services and API integration.
