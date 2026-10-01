# Integritree methodology: completed work and next steps

Study companion reviewed on **2026-10-01**. Return to the
[study index](STUDY_GUIDE.md), review [terms](STUDY_TERMS.md), or trace the
[code implementation](STUDY_CODE_GUIDE.md).

This guide explains the procedure already implemented and recorded. It does not
authorize another training run, change the frozen experiment, or claim that the
pending official test evaluation has happened.

## 1. What the research is trying to establish

The study compares two Random Forest models:

- **RF:** trained on the original imbalanced training partition.
- **RF-SMOTE:** trained on the same original partition after ordinary SMOTE adds
  synthetic fraud training vectors.

The comparison asks how SMOTE affects precision, recall, F1, MCC, and PR-AUC
(Average Precision), and whether paired classification error rates differ under
McNemar's test. Both models have matched forest settings and use the same
evaluation records and decision cutoff. Improvement from SMOTE is not assumed.

The receipt application demonstrates the trained models on confirmed screenshot
fields. Its operational success is a different question from model performance
on synthetic PaySim data or actual GCash fraud outcomes.

## 2. Current status and evidence

| Work | Status as of this review | Evidence / remaining boundary |
| --- | --- | --- |
| Source audit and preparation | Complete | Run `paysim_phase2_20260918`; split counts and provenance recorded. |
| Shared preprocessing | Complete | Eleven features; amount median/scaling fitted on original training only. |
| Paired model fitting | Complete for selection candidates | Saved paired artifacts and training audits. |
| Stage 1: SMOTE ratio | Complete | Selected 1:100 from eight candidates. |
| Stage 2: shared forest | Complete | Selected 100 trees, depth 10, minimum leaf size 1 from twelve configurations. |
| Stage 3: common cutoff | Complete | Selected 0.43 from 100 whole-percentage candidates on September 29, 2026. |
| Researcher application | Implemented | CSV validation, paired predictions/evaluation, requested SHAP, export, cleanup. |
| Receipt application | Implemented for recognized layouts | OCR, confirmation, mapping v2, paired inference/SHAP, export, cleanup. |
| OCR comparison | Completed local baseline | RapidOCR provisionally selected; small annotation sample limits conclusions. |
| Broader receipt acceptance | Pending | More unseen Pay Online/bank samples; Cash In/Out and merchant QR recognition remain pending. |
| Official held-out evaluation | Pending | Whole-tool readiness, explicit authorization, and reserved-test provenance required. |
| Official final-test SHAP/reporting | Pending with final evaluation | Application explanations already work; that does not mean final-test explanation/reporting is complete. |
| Thesis manuscript alignment | Pending review/application | Checklist items remain unchecked until actually applied to the manuscript. |

For execution status, use [VALIDATION_RUN_STATUS.md](VALIDATION_RUN_STATUS.md).
For current application behavior, use [RECEIPT_APPLICATION.md](RECEIPT_APPLICATION.md)
and [RESEARCHER_WORKFLOW.md](RESEARCHER_WORKFLOW.md). Older phase notes may describe
the state at the end of an earlier phase rather than today's state.

## 3. Step 1 — Identify and audit the dataset

The supplied source is `backend/data/raw/ps_raw.csv`, associated with the PaySim
source recorded in [METHODOLOGY.md](METHODOLOGY.md). Its exact local identity is:

```text
SHA-256: 16910f90577b0d981bf8ff289714510bb89bc71bff7d3f220f024e287e4eea6b
Records: 6,362,620
Raw columns: 11
Fraud: 8,213
Legitimate: 6,354,407
```

The eleven raw columns are `step`, `type`, `amount`, `nameOrig`, `oldbalanceOrg`,
`newbalanceOrig`, `nameDest`, `oldbalanceDest`, `newbalanceDest`, `isFraud`, and
`isFlaggedFraud`. A verified local fingerprint does not establish an independently
confirmed publisher release identifier.

The completed audit found no missing fields, invalid records, or exact duplicates.
All sixteen valid zero-amount transactions were retained. The source CSV was not
edited. Exact duplicate detection compared all eleven typed original columns;
it did not discard different transactions merely because their engineered
features happened to match.

**Why this matters:** later results must be traceable to a known source, and data
cleaning must not silently remove unusual but valid transactions. Record identity
combines the source fingerprint and original one-based data-row number.

Code to study: `audit_source()` in [data.py](../backend/src/integritree/ml/data.py).

## 4. Step 2 — Split before learning preprocessing

Shared stratified splits were created using two operations with seed 42:

1. Split the retained records into 80% training and 20% held out.
2. Split that held-out portion equally into validation and test.

| Partition | Total | Legitimate | Fraud | Use |
| --- | ---: | ---: | ---: | --- |
| Training | 5,090,096 | 5,083,526 | 6,570 | Fit preprocessing, RF, and RF-SMOTE. |
| Validation | 636,262 | 635,440 | 822 | Select ratio, shared forest settings, and common cutoff. |
| Test | 636,262 | 635,441 | 821 | Reserved for official finalized evaluation. |

Both models share the same row identities and labels within each partition.
Neither validation nor test receives SMOTE. No combined training/validation refit
is part of the current procedure.

**Why this matters:** fitting on a test distribution or generating synthetic
neighbors across partitions could leak information. Merely using different
filenames would not guarantee independence; saved membership and hashes provide
the evidence.

## 5. Step 3 — Build the eleven model features

Only five source attributes feed feature engineering: `step`, `type`, `amount`,
`nameOrig`, and `nameDest`. They are transformed; they are not passed as raw
columns directly into the forest.

| Order | Feature | Research source/formula |
| --- | --- | --- |
| 1 | `hour_of_day` | `(step - 1) % 24` |
| 2 | `day_of_week` | `((step - 1) // 24) % 7` |
| 3 | `type_CASH_IN` | 1 if type is CASH_IN, else 0 |
| 4 | `type_CASH_OUT` | 1 if type is CASH_OUT, else 0 |
| 5 | `type_DEBIT` | 1 if type is DEBIT, else 0 |
| 6 | `type_PAYMENT` | 1 if type is PAYMENT, else 0 |
| 7 | `type_TRANSFER` | 1 if type is TRANSFER, else 0 |
| 8 | `log_amount` | Natural logarithm `log(1 + amount)` |
| 9 | `is_zero_amount` | 1 if the completed amount equals zero, else 0 |
| 10 | `is_merchant_origin` | 1 if the source origin identifier starts with M, else 0 |
| 11 | `is_merchant_dest` | 1 if the source destination identifier starts with M, else 0 |

The four balance columns and `isFlaggedFraud` are excluded from predictors but
retained in source/audit information. `isFraud` is the separate target.

The hour/day are simulation-cycle positions. PaySim does not provide a verified
Monday or Manila calendar anchor in this CSV. For example, step 25 maps to hour 0
and day 1, without establishing a real Tuesday.

The inspected source has a constant-zero origin-merchant indicator, and the
destination-merchant indicator coincides with PAYMENT. These features were kept
under the approved schema, with the constant/redundant behavior acknowledged.

### Missing inputs and scaling

- Missing amount is permitted at the predictor boundary and uses the **original
  training median**. No missing-value replacements were needed for the supplied
  complete research file.
- Missing type uses all five type indicators equal to zero. There is no sixth
  model type column.
- Invalid/missing required time/entity information is rejected; `isFraud` is
  never guessed or imputed for evaluation.
- Min-Max scaling is fitted on training values for `log_amount`, `hour_of_day`,
  and `day_of_week` only. Binary indicators are left unchanged on real records.
- Saved scaling is applied to validation, test, and uploads without refitting
  and without clipping values outside training extrema.

**Why scale if the classifier is a forest?** Scaling is part of the approved
shared preprocessing and also controls distances used by SMOTE. It is not justified
as a requirement that every Random Forest inherently needs normalized inputs.

Code: [features.py](../backend/src/integritree/ml/features.py) and
[preprocessing.py](../backend/src/integritree/ml/preprocessing.py).

## 6. Step 4 — Fit paired models with controlled differences

```text
Original training rows → shared fitted preprocessing → training feature matrix
                                                    ├─ original rows → RF
                                                    └─ ordinary SMOTE → RF-SMOTE
```

SMOTE uses `k_neighbors=5` and seed 42. The model seed is also 42. Both forests
share tree settings, bootstrap behavior, Gini criterion, `max_features="sqrt"`,
and no class weighting. The intended experimental difference is the training
resampling branch, with shared hyperparameters chosen by the sequential protocol.

SMOTE operates on float64 encoded features. Fractions in category/merchant
indicators and time features are preserved, not rounded or filtered into more
realistic-looking transactions. Generated vectors remain fraud-labeled numeric
training examples. Their semantic realism is a limitation.

For the selected ratio 1:100, the resampled fraud count is approximately 1% of
the unchanged legitimate training count. That is not a fully balanced 1:1 dataset.
Use the selected candidate's `training_audit.json` for exact generated counts and
rounding; do not infer a precise count merely from the printed ratio.

Code: `resample_training()` and `train_models()` in
[training.py](../backend/src/integritree/ml/training.py).

## 7. Step 5 — Select settings in three validation stages

The selection keeps **both models**. It chooses shared experimental settings,
not a single model to retain as a winner.

| Stage | Candidates | Objective | Exact tie rule | Selected result |
| --- | --- | --- | --- | --- |
| 1: ratio | Fraud:legitimate 1:100, 1:50, 1:20, 1:10, 1:5, 1:3, 1:2, 1:1; reference forest 100/depth 10/leaf 1 | Highest RF-SMOTE validation AP | Smaller ratio | 1:100 |
| 2: forest | Trees {100,200} × depth {10,20} × leaf {1,10,50}: 12 matched configurations | Highest arithmetic mean AP across RF and RF-SMOTE | Shallower depth, fewer trees, larger leaf | 100/depth 10/leaf 1 |
| 3: cutoff | Exactly 0.01, 0.02, …, 1.00; no 0% candidate | Highest exact mean validation F1 across both models | Closest to 0.50, then higher threshold | 0.43 |

Freeze each stage before the next. Stage 3 reuses the fitted pair and its scores;
it does not fit another forest for each cutoff. Fraction-based comparison avoids
choosing a winner from rounded F1 displays. The cutoff is best among the tested
whole percentages, not necessarily every possible real-valued threshold.

The 0.50 classification checkpoint in Stages 1 and 2 is supplementary. It did
not select their winners and is not the final cutoff.

### What was recorded

Selection run: `paysim_three_stage_20260928_172539`.

Stage 1 completed September 28; Stages 2 and 3 completed September 29, 2026.
The authorized Stage 3 rerun finished without retraining. The recorded completion
has `completed_stage: 3`, `stage: validation_selected`, and `test_used: false`.

**Validation results at cutoff 0.43 — not final test results:**

| Metric | RF | RF-SMOTE |
| --- | ---: | ---: |
| Precision | 0.837209 | 0.427471 |
| Recall | 0.306569 | 0.405109 |
| F1 | 0.448798 | 0.415990 |
| MCC | 0.506284 | 0.415405 |
| Average Precision | 0.391528 | 0.381196 |

Mean F1 was approximately 0.432393935 over 636,262 validation records, including
822 labeled fraud. These rounded values come from
[the current run status](VALIDATION_RUN_STATUS.md); use its linked full-precision
outputs for calculations.

On this validation set, RF-SMOTE had higher recall but lower precision, F1, MCC,
and AP. This illustrates a trade-off; it does not establish final-test performance
or justify assuming that SMOTE improves every metric.

### Limitations to report honestly

The protocol was revised after validation inspection. Earlier Stage 3 outputs
were removed under the documented reset; preserved Stages 1–2 evidence was checked,
and review snapshots were regenerated. Do not describe the revised protocol as
historically preregistered.

Sequential freezing can miss ratio/forest interactions. Repeated use of one
validation set, one seed set, and prior inspection can influence selection.
Validation results are development evidence, not an unbiased final estimate.

Sources: [canonical protocol](THREE_STAGE_VALIDATION.md),
[run/reset record](VALIDATION_RUN_STATUS.md),
[selection code](../backend/src/integritree/ml/staged_selection.py), and
[threshold sweep](../backend/src/integritree/ml/threshold_search.py).

## 8. Step 6 — Evaluate and explain using the frozen policy

The metric/statistical implementation is complete. The official test execution
is still pending. Keep these two statements separate in presentations.

The evaluation policy is:

- Fraud is positive; classify using unrounded `score >= 0.43` for both models.
- Report precision, recall, count-based F1, MCC, and AP; accuracy is supplementary.
- Use the Average Precision convention for PR-AUC, with continuous scores retained.
- Report undefined metrics as unavailable with reasons.
- Compare metrics descriptively using signed symmetric percentage difference;
  use coefficient difference if either compared MCC is negative.
- Apply continuity-corrected McNemar at alpha 0.05 to paired correctness.
  Supplement with exact two-sided binomial results for 1–24 discordances.
  With zero discordances, report p=1 by convention and no statistic.

The [terms guide](STUDY_TERMS.md#3-evaluation-terms-and-formulas) gives formulas and
interpretation. The implementation is in
[evaluation.py](../backend/src/integritree/ml/evaluation.py) and its
[SQLite adapter](../backend/src/integritree/services/research_metrics.py).

### SHAP procedure

Interventional TreeSHAP explains each model's fraud probability. Both use a shared
uniform sample of 200 original-training records without replacement, seed 42.
The background is not fraud-balanced. A separate uniform sample of 1,000 evaluation
records is the global-summary policy; full-population reporting is explicit work.

The engine checks that baseline plus contributions reconstructs the probability
within 1e-6. The top positive contribution uses tolerance 1e-9. A representation
adapter handles tree/threshold details without changing the saved predictions.

The application explains requested researcher records, including legitimate
predictions, and attempts both explanations for a confirmed receipt. Report actual
coverage; do not describe a partly explained upload as a full-population SHAP study.
SHAP explains model behavior, not the cause of actual fraud.

## 9. Step 7 — Connect the research to the two application views

### Researcher CSV view

The application evaluates independently labeled compatible uploads using the saved
pair. It neither retrains nor verifies that a file is the reserved test partition.
Exports explicitly say `scope: uploaded_dataset` and
`held_out_membership_verified: false`.

The final research record must therefore establish held-out membership separately.
Demonstration CSVs and arbitrary labeled uploads do not become official test data
because the interface produces a confusion matrix.

### Receipt demonstration

The current application recognizes Express Send, Pay Online, and Bank Transfer
Complete layouts. After supported-image recognition, all five model categories
are selectable. This is broader category entry, not recognition of Cash In/Out
or merchant QR screenshots.

Mapping version `gcash_confirmed_v2` uses these editable role defaults:

| Category | Origin | Destination | Merchant indicators (origin/destination) |
| --- | --- | --- | --- |
| TRANSFER | Client | Client | 0/0 |
| PAYMENT | Client | Merchant | 0/1 |
| DEBIT | Client | Merchant | 0/1 |
| CASH_IN | Merchant | Client | 1/0 |
| CASH_OUT | Client | Merchant | 0/1 |

Users can override the model roles. Source layout observations remain separate
and immutable; known wallet/mixed-bank destination screens are rejected by the
layout checks. DEBIT's model-role default does not verify that a real bank is a
merchant. These are demonstration assumptions.

Receipt features use numerical PHP principal excluding fees, Manila hour, and
Monday=0 weekday. Their equivalence to PaySim's amounts and simulation time has
not been validated. Merchant-origin inputs can also fall outside the source
dataset's observed constant-zero origin flag. Shared feature shape alone does
not establish that the receipt population matches the training population.

OCR, confirmation, paired predictions, SHAP, ZIP export, and temporary cleanup
are connected. Receipt results omit Ground Truth, Outcome, and evaluation metrics.
The system does not verify image authenticity or prove a transaction fraudulent.

### What the OCR baseline establishes

The local inventory contained 146 images and 136 unique pixel groups. The benchmark
processed 144 eligible files with both engines, but only 21 screenshots had visual
field annotations: 13 development and 8 verification examples.

RapidOCR matched all required annotated fields on 12/13 development and 5/8
verification examples; Tesseract matched 6/13 and 3/8. These are small annotated
sample results, not accuracy estimates across all 144 processed files or all GCash
screens. The annotations were assistant-read rather than independently
double-annotated by human researchers.

RapidOCR was provisionally selected before verification. There was no OCR
fine-tuning or fraud-model retraining in this comparison. See the full
[OCR benchmark and limitations](RECEIPT_OCR_BENCHMARK.md).

## 10. What remains to do

These items describe pending work and study/review tasks; they do not initiate
execution. Existing frozen models, source data, and split identities must remain
traceable throughout.

### A. Finish application acceptance

- [ ] Review additional unseen supported Pay Online and bank-account screenshots.
- [ ] Collect and verify cash/QR examples before claiming those layouts supported.
- [ ] Review OCR annotation quality and near-duplicate/sample limitations with the
  researchers; document any later evaluation separately from the original baseline.
- [ ] Verify the complete user flow after final changes: upload, correction,
  confirmation, both results, explanations/retry, download, Clear, refresh, and expiry.
- [ ] Ensure scope/help text accurately describes supported layouts, editable
  roles, mapping assumptions, and distinct researcher/receipt retention behavior.
- [ ] Preserve software-check evidence separately from fraud-performance tables.

Recorded receipt verification includes 329 backend tests and later UI regression
checks, as described in [RECEIPT_APPLICATION.md](RECEIPT_APPLICATION.md).
Those are historical recorded checks; this documentation task did not rerun them.

### B. Conduct the official held-out evaluation after authorization

The recorded [researcher workflow](RESEARCHER_WORKFLOW.md) requires tool readiness
and explicit authorization for the final experiment. This is why it remains a
pending step rather than an automatic follow-on to documentation work.

1. Confirm readiness and record authorization for final evaluation.
2. Verify the frozen selected bundle, common cutoff, saved preprocessing, and
   dependencies against the completed selection evidence.
3. Establish that the submitted records are exactly the reserved 636,262-record
   test partition, with 821 fraud labels, using saved membership/provenance rather
   than counts alone. Preserve a traceable mapping if exporting a new CSV changes
   its file hash and application row identities.
4. Evaluate through the researcher tool as recorded in the workflow. Keep labels
   independent, use both fixed models on identical records, and retain full scores.
5. Download and preserve the whole-upload ZIP before backend shutdown/cleanup.
   It contains results, evaluation, metadata, and an offline report.
6. Verify record alignment, counts, confusion matrices, metric conventions, paired
   statistics, model/cutoff identity, and exported evidence.
7. Produce the planned explanation/report outputs with explicit coverage and
   sampling details. Do not imply the UI explained every uploaded transaction.
8. Report final-test findings and limitations without retuning on those results.

If final testing reveals a need for methodological changes, discuss and document
the implications for the evaluation design; do not silently tune against the test
set and keep calling it untouched.

### C. Update the paper and study materials

- [ ] Apply [THESIS_DOCUMENT_CHANGES.md](THESIS_DOCUMENT_CHANGES.md) to the manuscript,
  checking items only after reviewing the actual revised text.
- [ ] Correct F1/MCC equations and clearly identify AP as the PR-AUC convention.
- [ ] Explain mean tree probability, the 0.43 cutoff, and independent display bands.
- [ ] Report all selection candidates, objectives/tie rules, selected settings,
  and protocol revision history.
- [ ] Label validation and final-test tables distinctly; final-test tables remain
  pending until the authorized outputs exist.
- [ ] Discuss synthetic data, imbalance, SMOTE fractions, constant/redundant
  features, repeated validation use, single seeds, and receipt domain assumptions.
- [ ] Separate software verification, OCR extraction evidence, PaySim evaluation,
  and any future genuine GCash study.
- [ ] State conclusions from the measured comparison, including trade-offs or
  findings that do not favor SMOTE.

## 11. Questions to rehearse before a presentation or defense

1. What is controlled between RF and RF-SMOTE, and what differs?
2. Why are splitting and training-only preprocessing ordered this way?
3. Why do the source CSV and the model matrix both have eleven columns but different meanings?
4. Why retain all five categories if original fraud labels occur in only two?
5. Why preserve fractional SMOTE indicators, and what limitation follows?
6. Why use AP for Stages 1–2 and mean F1 for Stage 3?
7. What does 1:100 mean, and why is it not 1:1 balancing?
8. How was 43% selected, and what does “best” mean within that search?
9. What can and cannot be concluded from the current validation table?
10. Why does a passing browser/OCR test not establish fraud-detection performance?
11. How will the researchers prove a final uploaded CSV is the reserved test set?
12. What remains unfinished before final thesis conclusions can be written?

## 12. Supporting records

- [Methodology decision history](METHODOLOGY.md)
- [Canonical three-stage rules](THREE_STAGE_VALIDATION.md)
- [Completed selection and reset evidence](VALIDATION_RUN_STATUS.md)
- [Phase 4 evaluation/explanation implementation](PHASE4_IMPLEMENTATION.md)
- [Researcher operation and export scope](RESEARCHER_WORKFLOW.md)
- [Receipt application and acceptance evidence](RECEIPT_APPLICATION.md)
- [Current receipt mapping](RECEIPT_MAPPING.md)
- [OCR benchmark](RECEIPT_OCR_BENCHMARK.md)
- [Manuscript revision checklist](THESIS_DOCUMENT_CHANGES.md)
