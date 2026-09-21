# Integritree — Thesis Context and Confirmed Requirements

Last reviewed: 2026-09-22

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
Preprocessing was fitted on original training records only. Full PaySim baseline
RF/RF-SMOTE training is now complete and verified; see the Phase 3 decision record.

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

- PR-AUC numerical convention (AP versus trapezoidal); Equation 10 does not decide it.
- McNemar implementation and small-discordance handling. Signed comparison,
  undefined-value display and zero-discordance presentation are approved.
- Final RF configuration and cutoff selected by the approved future validation
  search; its procedure is settled, its winning settings are not known.
- SHAP background, output space, and computation coverage.
- Explainer/runtime details remain open; the score display is now 0..100 with
  approved communication bands, while internal scores remain 0..1.

The Phase 3 baseline is ordinary SMOTE at 1:1, k_neighbors=5, model/SMOTE seeds=42,
matching RF settings, and common 0.50 cutoff. The bounded shared-configuration
and common-cutoff search is documented in METHODOLOGY.md and follows Phase 4
metric implementation. No per-model thresholds are planned.

Receipt demonstration:

- Supported GCash receipt layouts and OCR engine.
- Mapping receipt fields to the trained feature meanings, including
  timing, amount units, transaction type, and merchant indicators.
- Handling incomplete, unreadable, and incompatible records.
- Manual-entry fallback.
- Upload retention and handling of personal receipt information.

Software:

- Dependency changes required for future SHAP/OCR work. Phases 1-3 use
  Python 3.12, FastAPI, NumPy, pandas, PyArrow, scikit-learn, imbalanced-learn,
  and joblib with a versioned lock.
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
- The 2026-09-22 audit passed 125 software tests. Completed Phase 2 full-output checks verified every
  row for alignment/valid features, file hashes, and preprocessing replay.
- Preparation and baseline training commands are implemented. Shared Python
  inference supports one or many structured records through saved models.
- Phase 3 and full PaySim baseline training are complete and verified. Both models
  are saved under backend/artifacts/paysim_phase3_baseline_20260920/. The successful
  retry followed an initial RAM preflight stop; the approved data/settings were unchanged.
- Evaluation, SHAP, batch-export CLI, prediction APIs, and OCR remain future work.
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
- The researchers authorized the scaffold, phased plan, and Phases 1-3 code.
  Phase 3 code and full PaySim baseline training are complete.
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

## 13. Current Phase 3 decision and execution status

Read the Phase 3 section of [METHODOLOGY.md](METHODOLOGY.md). It distinguishes the
implemented fixed baseline, approved but unexecuted validation search, and
unresolved evaluation choices. The Phase 2 bundle retains its original
configuration snapshot unchanged.

Phase 3 and full PaySim baseline training are complete. Run
paysim_phase3_baseline_20260920 contains both saved models and their provenance.
The initial memory preflight stop was resolved after unused applications were
closed. The successful retry used matching 100-tree/depth-20 models, all original
training rows, and the approved 1:1 SMOTE configuration.

The training audit confirms 5,076,956 synthetic fraud rows and 10,167,052 total
RF-SMOTE rows; no fractional categorical indicators arose in this particular run.
Artifact reloads, feature replay, paired inference consistency, and source/prepared
file integrity passed verification. See
backend/reports/paysim_phase3_baseline_20260920/verification.json.
Validation tuning, official test evaluation, and SHAP have not run. Do not describe
baseline artifacts or software verification as finalized performance findings.

Use [THESIS_DOCUMENT_CHANGES.md](THESIS_DOCUMENT_CHANGES.md) when aligning the
manuscript. Do not claim planned evaluation, tuning, SHAP, or GCash OCR is complete.

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
  Metric calculation is pending Phase 4, with explicit edge-case policies.
- GCash person-to-person receipt workflow only; TRANSFER selectable, other types
  visible but disabled, server-enforced. Exact names remain optional traceability
  information and never predictors. Receipt timing/amount mapping is experimental.

[The backend audit](BACKEND_ALIGNMENT_AUDIT.md) found the completed preparation and
baseline method aligned and documented future extensions: selected-run artifacts,
evaluation/SHAP, application results/query/export contracts, and a receipt adapter
using shared saved scaling. The plan now starts next with Phase 4. No new model
training or official performance result was produced by this review.
