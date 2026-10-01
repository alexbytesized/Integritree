# Thesis manuscript alignment checklist

Updated: 2026-10-01, aligned with the completed validation run and receipt mapping v2.

The Word reference documents remain unchanged. Apply the following corrections
when updating the manuscript; report numbers only from verified current outputs.
Unchecked items track manuscript review and application, not unfinished software.

The sole selection workflow is [three-stage validation](THREE_STAGE_VALIDATION.md):
SMOTE ratio by RF-SMOTE validation Average Precision, shared forest settings by
mean validation AP, then a common threshold by exact mean validation F1. Each
stage freezes before the next. Both models are retained; selection does not pick
a model winner. Stage 1 froze 1:100; Stage 2 selected 100 trees/depth 10/leaf 1.
Stage 3 completed on 2026-09-29 and selected **43% (0.43)** from exactly
1%, 2%, ..., 100% for run `paysim_three_stage_20260928_172539`.
All three selection stages are complete; official held-out test evaluation remains
pending. See [run status](VALIDATION_RUN_STATUS.md) for measured validation outcomes.

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
| [ ] | Add | SMOTE procedure | Ordinary SMOTE, eight candidate ratios from the protocol, k_neighbors=5, seed=42. Apply only to the original training branch for RF-SMOTE. Require at least k+1 fraud examples. RF uses the unaugmented training set. | training.py and canonical protocol; execution status recorded separately. |
| [ ] | Clarify limitation | SMOTE and categorical encoding | Convert encoded inputs to float64 before SMOTE and preserve interpolated fractions, including categorical indicators. Do not round, truncate, select argmax, or assign synthetic rows a real transaction type. Such rows are numeric training vectors, not realistic receipts. Their fraud target remains 1. | Inspect each new candidate audit; do not infer realism from numeric validity. |
| [ ] | Add | SMOTE sample counts | Report original, generated, and final class counts for each candidate from its new training audit. | Current run training_audit.json files. |
| [ ] | Add | Model training settings | Stage 1 used 100 trees/depth 10/leaf 1; Stage 2 compared the 12 shared settings in the protocol and selected 100 trees/depth 10/leaf 1. Keep other RF settings and seeds matched. | Canonical protocol and completed run evidence in VALIDATION_RUN_STATUS.md. |
| [ ] | Revise | Classification/risk-score description and equation | Define fraud score as the mean of individual trees' fraud-class probability estimates, using the class-1 column of predict_proba. A proportion of hard fraud votes is a different calculation. Internal score is 0..1. Do not describe it as a calibrated real-world probability without validation. | inference.py; a test demonstrates the difference from hard voting. |
| [ ] | Add | Classification threshold | Fraud when score >= 0.43 for the selected pair. The 0.50 checkpoint was supplementary in Stages 1 and 2; it is not the selected cutoff. The 43% result is optimal among the 100 tested percentages, not all possible thresholds. | Canonical protocol and completed Stage 3 outputs. |
| [ ] | Add | System architecture/reproducibility | Python scripts perform preparation and training. Shared inference loads both models plus fitted preprocessing, feature order, scoring policy, dataset/split fingerprints, software versions, and settings from a new run bundle. It does not retrain on application requests. | artifacts.py, inference.py, training.py. |
| [ ] | Clarify | Implementation/testing status | Separate synthetic software checks from completed research stages and official performance evaluation. | Current run status. |

## B. Describe three-stage validation

- [ ] State the eight ratios, reference forest, RF-SMOTE AP objective, and exact
  smaller-ratio tie rule for Stage 1.
- [ ] State the 12 matched forests, mean AP objective, and shallower/fewer-trees/
  larger-leaf tie order for Stage 2.
- [ ] State Stage 3's 100 thresholds (1%, 2%, ..., 100%); maximize
  exact mean F1, breaking ties by proximity to 0.50 and then the higher threshold.
- [ ] Freeze each stage before the next. Keep both models, shared settings, and one
  cutoff. Classification metrics at 0.50 supplement Stages 1 and 2 only.
- [ ] Report all candidates and acknowledge sequential ratio/forest interactions,
  repeated validation use, one seed set, and prior inspection of this validation
  set. Do not claim a global optimum or historical preregistration.
- [ ] Report completed selection: 1:100 SMOTE ratio, 100 trees/depth 10/leaf 1,
  and shared cutoff 0.43. Stage 3 completed on 2026-09-29 without retraining.
  Official test evaluation remains pending and requires separate authorization;
  there is no combined training/validation refit.
- [ ] Disclose that the threshold protocol was revised after validation inspection.
  Use the current 1%-100% grid results and the reset/preservation record; do not
  present the revised protocol as historically preregistered.

Use [the canonical protocol](THREE_STAGE_VALIDATION.md) for the exact settings.

## C. Align evaluation text with Phase 4

Metric/statistical code is implemented and current validation outputs are available.
Official thesis test results remain pending; label validation results accordingly.
The manuscript corrections remain unchecked until reviewed and applied.

| Done | Action | Location | Change needed |
| --- | --- | --- | --- |
| [ ] | Correct | Equation 8, F1 | Use F1=2PR/(P+R)=2TP/(2TP+FP+FN). The current expanded expression has incorrect repeated terms/factors. Specify zero-denominator behavior with Phase 4. |
| [ ] | Correct | Equation 9, MCC | Replace TR with TP in the numerator: TP*TN-FP*FN. Denominator is sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN)). Specify degenerate-case handling with Phase 4. |
| [ ] | Clarify | Equation 10, PR-AUC | Average Precision (AP) was explicitly approved on 2026-09-23. Clarify that AP is the chosen numerical convention and label results accordingly. The existing integral alone did not uniquely select AP; Phase 4 implements AP; specify the non-trapezoidal convention. |
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
- [ ] **Add the experimental demonstration scope:** GCash app screenshots for
  Express Send/TRANSFER, over-the-counter cash-in/CASH_IN, over-the-counter cash-out/
  CASH_OUT, wallet-funded merchant QR or Pay Online/PAYMENT, and bank-account transfer/DEBIT. Include local
  extraction, confirmation/completion, mapping, and both models' predictions.
  This scope supersedes person-to-person-only support. The connected application
  currently recognizes Express Send, Pay Online, and Bank Transfer Complete layouts,
  with local RapidOCR, editable confirmation, paired predictions, SHAP waterfalls,
  ZIP download, and cleanup. Cash In/Out and merchant QR screenshot recognition
  remain pending. See [application status](RECEIPT_APPLICATION.md).
- [ ] **Distinguish category coverage from fraud coverage:** all five types are
  included in the dataset/models, but original fraud labels occur only in TRANSFER
  and CASH_OUT. All five categories are selectable after a supported screenshot is
  recognized; this does not enable recognition of five screenshot categories.
  Neither SMOTE nor category selection validates real GCash fraud detection.
- [ ] **Clarify receipt behavior:** the models operate on structured transaction
  features, not receipt pixels. The system does not verify image authenticity,
  identify a proven scam category, or confirm that a transaction was actually fraud.
- [ ] **Clarify the two views:** researcher evaluation requires compatible,
  independently labeled held-out data; individual prediction has no accuracy,
  precision, recall, or other evaluation results without independent labels.
- [ ] **Document implemented SHAP:** interventional fraud-probability TreeSHAP,
  shared uniform sample of 200 original-training records without replacement,
  seed 42, reconstruction tolerance 1e-6 and positive tolerance 1e-9. Record the
  1,000-record global sampling policy and report actual explanation coverage separately. Explain the prediction-equivalent representation adapter in the technical
  implementation section; it does not alter the model or explanation target.
- [ ] **Align the approved SHAP presentation:** top risk-increasing contributor
  even for legitimate predictions, deterministic explanations, and waterfall-only
  expanded details. Numerical table is suggestion-only. Document any explanation
  coverage broader than fraud predictions, and distinguish partial coverage.
- [ ] **Document communication bands and labels:** Predicted Fraud / Predicted
  Legitimate, score 100*p, Minimal [0,20), Low [20,40), Moderate [40,60), High [60,80),
  Critical [80,100]. These bands are not the binary decision threshold or validated
  real-world fraud probabilities.
- [ ] **Align result presentation and exports:** Original Inputs / Derived Inputs,
  known ground truth and model/outcome filters for researcher records, and researcher
  ZIP results/evaluation/provenance export. Receipt results omit Ground Truth,
  Outcome, and evaluation metrics. Keep Research Scope and Limitations text aligned
  with the implementation. No extra correctness badge, on-screen run panel, or
  on-screen evaluation metadata panel is required by the agreed UI.
- [ ] **Record settled receipt decisions:** Manila hour/Monday=0 weekday, numeric
  PHP principal excluding fees, post-upload correction/completion and confirmation,
  optional reference, and local OCR. Proceed explicitly confirms the editable
  fields; separate review/funding checkboxes are not required. No image-free manual
  entry is provided. The PHP and calendar mappings remain demonstration assumptions,
  not verified equivalence to PaySim amounts or simulation time.
- [ ] **Document implemented retention:** leaving the receipt flow or choosing Clear
  invalidates access immediately; active writers finish before physical deletion.
  The last browser connection disconnect starts a 30-second reconnection grace
  period; refresh preserves the session when it reconnects in time. Shutdown removes
  this process's session files and startup removes abandoned sessions. Network loss
  or browser suspension can expire the session. Development samples and downloaded
  ZIPs are retained separately; failed deletion is retried. See
  [retention details and limits](RECEIPT_APPLICATION.md#download-and-retention).
- [ ] **Record the local OCR baseline:** RapidOCR provisionally selected after
  development comparison and frozen verification; see [measured limits](RECEIPT_OCR_BENCHMARK.md).
  This is separate from fraud-model performance, with only 21 visually annotated images.
- [ ] **Document receipt mapping v2:** `gcash_confirmed_v2` derives merchant flags
  from independently editable Client/Merchant roles. Default origin/destination
  flags are TRANSFER 0/0, PAYMENT 0/1, DEBIT 0/1, CASH_IN 1/0, and CASH_OUT 0/1.
  Changing category resets role defaults; users can then override them. These are
  demonstration model-role assumptions, not verified recipient identities.
  Server-owned source/layout observations remain separate and immutable; known
  wallet/mixed-bank destinations are rejected by layout checks. All five categories
  may be confirmed within a supported upload, but cash/QR layout acceptance remains
  pending. See [mapping implementation](RECEIPT_MAPPING.md).
- [ ] **Document implemented receipt contracts and exports:** one PNG/JPEG up to
  10 MiB and 20 million decoded pixels; session-owned image jobs and HTTP/WebSocket
  routes; revision-bound confirmation, predictions, explanations, and downloads.
  Reconfirmation invalidates older derived results. The individual ZIP contains
  confirmed inputs, paired results, explanations, metadata, and computed waterfall
  SVGs; it excludes the original image and evaluation metrics. See
  [API and file contracts](RECEIPT_APPLICATION.md#http-contract).
- [ ] **Separate integration evidence from acceptance and performance:** synthetic
  tests, browser checks, and a local real-image OCR/model/SHAP/ZIP smoke test verify
  implemented behavior. Broader unseen Pay Online and bank-account samples and
  cash/QR collection remain outstanding. These checks do not establish real GCash
  fraud-detection accuracy or constitute official held-out thesis evaluation.

## Suggested replacement passages for review

**Training and scoring (implemented behavior):**

> The experiment compares Random Forest models with identical classifier settings
> and random seeds. The control is fitted to the original training split. For the
> experimental branch, ordinary SMOTE is applied only to the training features to
> use the selected 1:100 fraud-to-legitimate ratio. Both selected forests use
> 100 trees, maximum depth 10, and minimum leaf size 1. Fractional values produced in
> encoded indicators are retained as synthetic numerical training inputs and
> acknowledged as a limitation. Each model's fraud risk score is the mean of its
> trees' fraud-class probability estimates. Both models classify a transaction as
> fraud when this score is at least the shared Stage 3 cutoff of 0.43.

**Validation and final testing:**

> Three sequential validation stages selected the SMOTE ratio by RF-SMOTE Average
> Precision, matched forest settings by mean Average Precision, and a common
> threshold by exact mean F1. Each decision was frozen before the next stage.
> Selection used validation only and completed on September 29, 2026. The selected
> cutoff of 0.43 maximized mean F1 among the 100 tested percentage thresholds.
> The threshold protocol was revised after validation inspection. Both fitted
> models and the cutoff are fixed; separately authorized evaluation on the
> untouched test set remains pending.

Use past tense for completed selection and retain pending language for official
test evaluation until verified outputs exist. Include the search rules and
limitations from the checklist, rather than relying on these short passages alone.

## Supporting implementation and technical references

- [Methodology decision record](METHODOLOGY.md)
- [Backend implementation plan](BACKEND_IMPLEMENTATION_PLAN.md)
- [Backend usage](../backend/README.md)
- [Current validation run status](VALIDATION_RUN_STATUS.md)
- [Connected receipt application and verification](RECEIPT_APPLICATION.md)
- [Confirmed receipt mapping v2](RECEIPT_MAPPING.md)
- [Researcher application workflow](RESEARCHER_WORKFLOW.md)
- [Feature implementation](../backend/src/integritree/ml/features.py)
- [Training implementation](../backend/src/integritree/ml/training.py)
- [Inference implementation](../backend/src/integritree/ml/inference.py)
- [SMOTE parameters](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)
- [Random Forest probabilities](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [Average Precision and its difference from trapezoidal integration](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)
- [Preprocessing leakage](https://scikit-learn.org/stable/common_pitfalls.html)


## E. Phase 4 additions for manuscript review

- [ ] Define undefined precision/recall/F1/MCC explicitly. Return unavailable with
  a reason for zero denominators; direct-count F1 is valid zero when its denominator
  is positive. AP is unavailable with no positives and equals 1 with all positives.
- [ ] Keep continuity-corrected McNemar primary, with the supplementary exact
  two-sided binomial check for 1-24 discordances. Zero discordances use p=1 by
  convention and an unavailable statistic. Report the primary and supplementary
  methods distinctly rather than choosing whichever p-value is smaller.
- [ ] Specify the fixed background sample and global explanation sample separately.
  Both are uniform without replacement with seed 42. Neither is fraud-balanced.
  Background is original training only; explanation targets are evaluation records.
- [ ] Describe SHAP 0.52.0 and the equivalent internal tree representation needed
  for float32 thresholds/large node counts. Cite the algorithm/library and record
  reconstruction and exhaustive-coalition verification; do not describe this as
  retraining, approximate explanations, or edited saved models.
- [ ] State that the real three-stage search is complete and the selected bundle
  uses a frozen common cutoff of 0.43. Do not report .50 as validation-optimal.
- [ ] Final thesis performance tables must come from held-out testing after
  completed selection and separate authorization. Metrics from all three selection
  stages are validation development evidence, not official test results.
- [ ] Distinguish software tests from fraud-detection performance studies and
  evaluation on genuine GCash transactions.
