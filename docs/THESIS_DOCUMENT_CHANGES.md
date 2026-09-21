# Thesis manuscript alignment checklist

Last reviewed: 2026-09-20

This is a checklist for researcher/adviser review, not an edited manuscript.
Source: `C:/Users/Carpicorn/Downloads/Thesis/COSC-305_CS-Thesis-Writing-1/Post-Proposal-Defense_Revision/Thesis-Proposal-Revised.docx`.
The original Chapter 3 wording and equations were inspected. Section names and
equation numbers below identify locations; page numbers may change during editing.

**Implementation status:** Phases 1-3, full PaySim preparation, and baseline
RF/RF-SMOTE training are complete and verified. Run
paysim_phase3_baseline_20260920 contains both trained models and provenance.
The initial RAM preflight stop was resolved after unused applications were closed;
the successful retry preserved the approved settings and all training records.
Validation tuning, official test evaluation, SHAP, prediction API integration, and
OCR remain future work. Trained baseline models are not final performance results.

Use the checkboxes when the researchers actually update and review the manuscript.
All items remain unchecked because no manuscript edits were made.
“Implemented” means code exists; full-dataset execution is identified separately.

## A. Align descriptions with implemented behavior

| Done | Action | Location | Required alignment | Status/evidence |
| --- | --- | --- | --- | --- |
| [ ] | Clarify | Chapter 3, dataset description | Identify the supplied PaySim file, source URL, SHA-256, 6,362,620 records, 11 raw columns, 8,213 fraud and 6,354,407 legitimate records. Do not claim a publisher release identifier that was not established. | Full preparation completed; METHODOLOGY.md and preparation audit. |
| [ ] | Clarify | Data cleaning | Exact duplicates are checked using all 11 typed original columns before splitting; retain the first occurrence. Do not deduplicate by engineered features. Report zero exact duplicates in this supplied file. Preserve the raw CSV. | Implemented and executed. |
| [ ] | Revise/clarify | Preprocessing sequence and diagrams | Separate deterministic feature calculations from fitted transformations. Create split membership before learning amount imputation or scaling parameters. Fit them on original training rows only, before SMOTE; reuse on validation, test, and inference. | preparation.py, preprocessing.py. |
| [ ] | Add | Missing/invalid data handling | Missing amount uses the training median; missing type uses all-zero type indicators. There is no sixth model type column. Missing/invalid target or required step/entity attributes cause rejection; do not invent them. The supplied file needed no missing-value replacements. | features.py, data.py, preprocessing.py. |
| [ ] | Clarify | Feature selection | Derive features from step, type, amount, nameOrig, nameDest; none of these raw columns directly enters RF. Exclude four balances and isFlaggedFraud; keep isFraud separate as the target. Keep excluded columns in source/audit data. | FEATURE_COLUMNS and SOURCE_COLUMNS in features.py. |
| [ ] | Add | Feature engineering | State all 11 final features and their order, as listed in METHODOLOGY.md. Keep all five transaction types. Retain the constant origin-merchant indicator and destination-merchant/payment redundancy; acknowledge these observations rather than silently removing features. | Implemented; full source inspected. |
| [ ] | Clarify | Timing features | hour_of_day=(step-1)%24; day_of_week=((step-1)//24)%7. These are simulation-cycle positions; do not label day 0 as a verified Monday or treat PaySim time as an established real GCash calendar mapping. | Approved convention implemented. |
| [ ] | Add | Normalization | Apply Min-Max scaling to log_amount, hour_of_day, and day_of_week using training extrema only. Binary indicators remain 0/1 on real rows. Do not clip future values to the training range or refit on uploaded records. | preprocessing.py. |
| [ ] | Clarify | Splitting | Stratified 80/10/10 implemented as 80/20, followed by a 50/50 split of the held-out 20%, both with seed 42. Shared row identities keep outputs paired. Report train=5,090,096 (6,570 fraud), validation=636,262 (822 fraud), test=636,262 (821 fraud). | Full preparation completed. |
| [ ] | Add | SMOTE procedure | Ordinary SMOTE, minority:majority ratio 1:1, k_neighbors=5, seed=42. Apply only to the original training branch for RF-SMOTE. Require at least k+1 fraud examples. RF uses the unaugmented training set. | training.py implemented; full baseline training completed. |
| [ ] | Clarify limitation | SMOTE and categorical encoding | Convert encoded inputs to float64 before SMOTE and preserve interpolated fractions, including categorical indicators. Do not round, truncate, select argmax, or assign synthetic rows a real transaction type. Such rows are numeric training vectors, not realistic receipts. Their fraud target remains 1. | Implemented and verified. Full-run audit found zero fractional categorical indicators/mixed type rows; the general limitation remains. |
| [ ] | Add | SMOTE sample counts | Report the confirmed 5,076,956 additional fraud samples and 10,167,052 total RF-SMOTE training rows, with 5,083,526 in each class. These are training counts, not model-performance findings. | Confirmed by training_audit.json in the completed baseline run. |
| [ ] | Add | Model training settings | Shared baseline: 100 trees, max_depth=20, min_samples_split=2, min_samples_leaf=1, max_features=sqrt, bootstrap=True, class_weight=None, criterion=gini, model seed=42. Both models use matching settings; SMOTE is the intended training difference. | experiment.yaml, training.py. |
| [ ] | Revise | Classification/risk-score description and equation | Define fraud score as the mean of individual trees' fraud-class probability estimates, using the class-1 column of predict_proba. A proportion of hard fraud votes is a different calculation. Internal score is 0..1. Do not describe it as a calibrated real-world probability without validation. | inference.py; a test demonstrates the difference from hard voting. |
| [ ] | Add | Classification threshold | Baseline predicts fraud when score >= 0.50, including exact ties. Both models use the same cutoff. Baseline is not described as optimal or finalized after tuning. | Implemented. |
| [ ] | Add | System architecture/reproducibility | Python scripts perform preparation and training. Shared inference loads both models plus fitted preprocessing, feature order, scoring policy, dataset/split fingerprints, software versions, and settings from a new run bundle. It does not retrain on application requests. | artifacts.py, inference.py, training.py. |
| [ ] | Clarify | Implementation/testing status | Distinguish synthetic software tests, completed full baseline training, and artifact/inference consistency checks from future thesis performance evaluation. The memory issue was resolved on retry without reducing data or changing settings. | Full run and verification completed; see backend README and the saved verification report. |

## B. Record approved future tuning without claiming it has run

The researchers approved the following protocol for implementation after Phase 4's
metric code is available. It is not executed by the current fixed-baseline command.

- [ ] **Add to Model Validation:** candidate RF settings are (trees, depth):
  (100,10), (100,20), (200,10), (200,20); other settings remain fixed and matched.
  Evaluate both models for each candidate at the common 0.50 cutoff.
- [ ] **Add the selection rule:** select one shared RF configuration by the highest
  mean of RF validation F1 and RF-SMOTE validation F1. Exact ties prefer shallower
  trees, then fewer trees. Report individual model results as well as the mean.
- [ ] **Add common-threshold tuning:** after selecting the shared RF configuration,
  evaluate 0.05, 0.10, ..., 0.95 on the validation set. Select the common threshold
  giving the highest mean F1 across the pair. Exact ties prefer closest to 0.50,
  then the higher cutoff. This need not maximize each model's individual F1.
- [ ] **Clarify tuning limits:** define the search before inspecting its results;
  keep settings and cutoff common. Do not promise that a small search guarantees
  absence of overfitting or finds the global optimum.
- [ ] **Clarify order:** implement metrics, tune on validation, freeze models and
  cutoff, then perform official test evaluation. Do not retrain on combined
  training+validation data unless that separate methodology change is approved.

## C. Correct or clarify evaluation text before Phase 4

These corrections are discussed requirements, not a claim that metric/statistical
code is already implemented.

| Done | Action | Location | Change needed |
| --- | --- | --- | --- |
| [ ] | Correct | Equation 8, F1 | Use F1=2PR/(P+R)=2TP/(2TP+FP+FN). The current expanded expression has incorrect repeated terms/factors. Specify zero-denominator behavior with Phase 4. |
| [ ] | Correct | Equation 9, MCC | Replace TR with TP in the numerator: TP*TN-FP*FN. Denominator is sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN)). Specify degenerate-case handling with Phase 4. |
| [ ] | Clarify | Equation 10, PR-AUC | The integral gives a general area definition; it does not select Average Precision or trapezoidal integration. Agree on a numerical convention in Phase 4 and explicitly name it. Do not claim AP is already fixed or that the current equation uniquely mandates trapezoidal integration. |
| [ ] | Revise | Data Analysis introduction; Model Validation | A single confusion matrix supplies precision, recall, F1, MCC, and accuracy. PR-AUC needs ground truth and continuous fraud scores across thresholds. Save/use scores as well as predicted labels. |
| [ ] | Clarify | Chapter 2 metric review | The probability that a positive example outranks a negative describes ROC-AUC, not PR-AUC. Distinguish the two wherever the term AUC is used. |
| [ ] | Revise (approved 2026-09-22) | Percentage difference / Equation 11 | Replace absolute percentage difference with signed symmetric percentage difference: 100*(S-B)/((S+B)/2), where S=RF-SMOTE and B=benchmark RF. Positive favors RF-SMOTE; negative favors benchmark RF. For nonnegative scores both zero, report N/A; if either MCC is negative, report S-B in coefficient units instead. Undefined inputs produce N/A with a reason. This descriptive comparison does not establish statistical significance. Align method labels, help, configuration, and exports; do not rewrite saved historical configurations. |
| [ ] | Revise | McNemar discussion; interpretation of SOP 3 | McNemar on paired correct/incorrect predictions tests equality of classification error rates. It does not separately establish significance for precision, recall, F1, MCC, or PR-AUC. A significant difference can favor either model. |
| [ ] | Correct wording | Hypothesis decision rule | For p >= 0.05, use “fail to reject the null hypothesis,” not “accept.” Nonsignificance does not prove equivalence. |
| [ ] | Verify/clarify | Equation 12 and McNemar implementation | Check mathematical rendering of the absolute difference in the continuity-corrected statistic. Reconcile the exact formula and behavior for zero/few discordant pairs with the selected Phase 4 implementation; do not silently replace the manuscript's test. |
| [ ] | Clarify | Model testing/results | Use the same untouched held-out test records for both finalized models. Training predictions or arbitrary labeled uploads are not the official thesis test evaluation. |

## D. Scope, interface, and remaining decisions

- [ ] **Clarify Chapters 1 and 3:** PaySim is synthetic; its held-out evaluation
  does not establish performance on actual GCash/Maya transaction databases.
- [ ] **Add the experimental demonstration scope:** initial GCash person-to-person
  receipt upload, extraction, user confirmation/completion, input mapping, and
  both models' predictions. This is planned work, not currently implemented.
- [ ] **Clarify receipt behavior:** the models operate on structured transaction
  features, not receipt pixels. The system does not verify image authenticity,
  identify a proven scam category, or confirm that a transaction was actually fraud.
- [ ] **Clarify the two views:** researcher evaluation requires compatible,
  independently labeled held-out data; individual prediction has no accuracy,
  precision, recall, or other evaluation results without independent labels.
- [ ] **Finalize before claiming SHAP implementation:** explained output space,
  background selection/size, computation coverage, and additivity checks.
  Align the explanation target with the implemented fraud score.
- [ ] **Align the approved SHAP presentation:** top risk-increasing contributor
  even for legitimate predictions, deterministic explanations, and waterfall-only
  expanded details. Numerical table is suggestion-only. Document any explanation
  coverage broader than fraud predictions, and distinguish partial coverage.
- [ ] **Document communication bands and labels:** Predicted Fraud / Predicted
  Legitimate, score 100*p, Minimal [0,20), Low [20,40), Moderate [40,60), High [60,80),
  Critical [80,100]. These bands are not the binary decision threshold or validated
  real-world fraud probabilities.
- [ ] **Align result presentation and exports:** Original Inputs / Derived Inputs,
  known ground truth, model/outcome filters, ZIP results/evaluation/provenance export,
  and Research Scope and Limitations. No extra correctness badge, on-screen run
  panel, or on-screen evaluation metadata panel is required by the agreed UI.
- [ ] **Finalize receipt mapping before demonstration:** amount units, time
  convention, transaction type, entity indicators, missing/unreadable fields,
  manual fallback, and receipt retention. Do not fabricate unavailable inputs.

## Suggested replacement passages for review

**Training and scoring (implemented behavior):**

> The baseline compares Random Forest models with identical classifier settings
> and random seeds. The control is fitted to the original training split. For the
> experimental branch, ordinary SMOTE is applied only to the training features to
> target equal fraud and legitimate sample counts. Fractional values produced in
> encoded indicators are retained as synthetic numerical training inputs and
> acknowledged as a limitation. Each model's fraud risk score is the mean of its
> trees' fraud-class probability estimates. Both models classify a transaction as
> fraud when this score is at least 0.50 in the baseline experiment.

**Validation and final testing (approved future procedure):**

> A predefined, limited validation search will select shared classifier settings
> and a common decision threshold using the mean F1 score of the two models.
> Selection will use the validation split only. The final settings and threshold
> will be fixed before both models are evaluated on the shared held-out test set.

Adjust tense when the full experiment has actually run. Include the numeric
settings and search rules from the checklist, rather than relying on these short
passages alone.

## Supporting implementation and technical references

- [Methodology decision record](METHODOLOGY.md)
- [Backend implementation plan](BACKEND_IMPLEMENTATION_PLAN.md)
- [Backend usage and current run status](../backend/README.md)
- [Feature implementation](../backend/src/integritree/ml/features.py)
- [Training implementation](../backend/src/integritree/ml/training.py)
- [Inference implementation](../backend/src/integritree/ml/inference.py)
- [SMOTE parameters](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)
- [Random Forest probabilities](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [Average Precision and its difference from trapezoidal integration](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)
- [Preprocessing leakage](https://scikit-learn.org/stable/common_pitfalls.html)
