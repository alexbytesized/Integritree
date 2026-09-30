# Methodology

The sole selection workflow is [three-stage validation](THREE_STAGE_VALIDATION.md):
SMOTE ratio by RF-SMOTE validation Average Precision, shared forest settings by
mean validation AP, then a common threshold by exact mean validation F1. Each
stage freezes before the next. Both models are retained; selection does not pick
a model winner. Stage 1 froze 1:100; Stage 2 selected 100 trees/depth 10/leaf 1.
Stage 3 uses exactly 1%, 2%, ..., 100% as threshold candidates. See
[run status](VALIDATION_RUN_STATUS.md) for measured outcomes.

## Confirmed constraints

Compare RF trained on the original training distribution with RF-SMOTE trained
on an augmented version of the same training set. Use stratified 80/10/10
training/validation/testing splits and shared validation/test records.
Fit learned transformations on training only and apply SMOTE only to training.

Keep `isFraud` separate from predictors. Exclude `oldbalanceOrg`,
`newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`, and `isFlaggedFraud`
from predictor inputs. Preserve valid unusual behavior, including zero amounts.

Evaluate precision, recall, F1, MCC, and PR-AUC on the held-out test set.
McNemar at alpha 0.05 concerns paired classification errors; it does not establish
the significance of each metric. Improvement from SMOTE is not assumed.

## Local dataset identified

The researchers supplied `backend/data/raw/ps_raw.csv` from
[the cited PaySim dataset](https://www.kaggle.com/datasets/ealaxi/paysim1/data).
The fingerprint identifies the exact local file; a publisher release identifier
was not independently established.

| Property | Inspection result |
| --- | --- |
| Size | 493,534,783 bytes |
| Data rows | 6,362,620 |
| Columns | 11 |
| Fraud / legitimate | 8,213 / 6,354,407 |
| Fraud share | Approximately 0.129082% |
| Step range | 1 to 743 |
| SHA-256 | `16910f90577b0d981bf8ff289714510bb89bc71bff7d3f220f024e287e4eea6b` |

The structural inspection found no missing/blank fields, invalid/nonfinite numeric
values, or negative amounts/balances. It did **not** check exact duplicate rows.
The file remains unchanged. Phase 2 now implements a reproducible audit and
verifies file identity before producing prepared datasets. The inspection above
describes the earlier structural review; the full-run audit is recorded separately.

| Raw column(s) | Intended treatment |
| --- | --- |
| `step` | Source for timing features using the approved simulation convention below. |
| `type` | Source for the five transaction-type indicators. |
| `amount` | Source for log amount and zero-amount indicator. |
| `nameOrig`, `nameDest` | Source for entity-type indicators; not approved as raw categorical model features. |
| Four balance columns | Retain in the raw source; exclude from predictors. |
| `isFlaggedFraud` | Retain in the raw source; exclude from predictors. |
| `isFraud` | Ground truth only. |

All origin identifiers have the C prefix: the proposed origin-merchant feature
would be constant zero. Destination M prefixes coincide with PAYMENT records:
the proposed destination-merchant feature would duplicate that type indicator.
These are observations, not permission to remove manuscript features.

Fraud labels occur in TRANSFER and CASH_OUT only. This does not authorize filtering
the research dataset to those types. All five categories are present and supported
by the trained models. CASH_IN, PAYMENT, and DEBIT contain no original fraud-positive
examples in this dataset; this reflects the synthetic scenario/label coverage, not
an absence of real-world fraud. SMOTE does not create missing real-world evidence
or validate fraud detection for those categories. Receipt input coverage must not
be reported as real GCash fraud-detection performance. Sixteen records have zero
amounts; keep valid unusual transactions. There is no calendar date/timezone anchor in the CSV, so
derived day/hour features must have an explicit simulation-time interpretation.

## Phase 1 software decisions

Use Python 3.12, FastAPI, Pydantic validation, and YAML experiment configuration.
The development dependencies are pinned in `backend/requirements-dev.lock`.
Training scripts and application services will import one shared Python package.

The raw record contract covers all 11 fields after numeric CSV parsing.
`PredictorInput` contains only `step`, `type`, `amount`, `nameOrig`,
and `nameDest`. These are source attributes for shared feature engineering,
not the final engineered model matrix or a GCash receipt mapping.

Identify a dataset record by the exact file's SHA-256 plus its one-based original
data-row number, excluding the CSV header. Assign it before cleaning and preserve
it through splitting and both model branches. Reordering the source file changes
its identity intentionally. Individual-request IDs will be assigned by the later
application workflow.

Result contracts carry both models, the run ID, predicted labels, and risk scores.
The internal score field is bounded to 0..1. Subsequent decisions fix mean-tree
fraud probability and a 0..100 display scale with communication bands; this does
not establish real-world calibration. SHAP contracts name the output space but
do not yet compute explanations or verify additivity.

## Approved Phase 2 preparation decisions

The researchers explicitly accepted all six recommendations with "Proceed with
all your recommendations." These decisions authorize the preparation settings;
Phase 2 processing is now implemented. The manuscript
has not been edited. Clarifying the simulation-time interpretation in Chapter 3
remains a documentation follow-up.

1. **Feature schema:** retain all 11 manuscript features in this model order:
   `hour_of_day`, `day_of_week`, `type_CASH_IN`, `type_CASH_OUT`, `type_DEBIT`,
   `type_PAYMENT`, `type_TRANSFER`, `log_amount`, `is_zero_amount`,
   `is_merchant_origin`, `is_merchant_dest`. Use the five original source
   attributes to derive them, then exclude those original columns from the model
   matrix. Preserve source attributes and identity in audit/prepared records.
   Retain and document both merchant indicators despite their constant/redundant
   behavior in the inspected file. Keep all five transaction types in the study.

2. **Simulation time:** use `hour_of_day = (step - 1) % 24` and
   `day_of_week = ((step - 1) // 24) % 7`. Step 1 means simulated hour 0/day 0;
   step 25 means hour 0/day 1. Days 0..6 are positions in a simulated seven-day
   cycle, not verified Monday..Sunday labels. This is an agreed convention, not
   evidence of the dataset's calendar origin or a verified GCash time mapping.
   Configuration name: `simulation_step_1_hour_0_day_0`.

3. **Exact duplicates:** compare all 11 original columns before splitting,
   retain the first source occurrence, and record removed-row counts and source
   identities. Do not deduplicate on engineered features: different transactions
   can share the same feature values. The full Phase 2 audit found zero exact
   duplicates, so all source records were retained. Never modify the raw CSV.

4. **Missing and invalid values:** apply training-only numerical medians where
   meaningful and explicitly represent missing categories as `unknown`. Never
   impute `isFraud`. Stop and report malformed records or missing time/entity
   information instead of guessing feature values. Keep valid unusual behavior,
   including zero amounts. The inspected source had no missing values.

   Implementation details consistent with this policy: a missing amount can be
   filled using the original training amounts' median before deriving amount
   features; an all-missing training amount column must fail clearly. A missing
   transaction type is retained as `unknown` in audit data and represented by
   zeros in all five type indicators, preserving the approved 11-feature schema.
   Nonempty unrecognized types are invalid, not silently treated as missing.
   Missing step, origin/destination identity, or target blocks research
   preparation with an audit error. No guessed merchant flag or extra unknown
   feature is introduced. Phase 2 has an ingestion layer that handles permitted
   missing inputs separately from the strict Phase 1 complete-record contracts.

5. **Normalization:** use training-fitted Min-Max scaling on `log_amount`,
   `hour_of_day`, and `day_of_week`; leave binary features at 0/1. Save/reuse
   exactly the same transformation for both model branches and other splits.
   Do not clip transformed values outside the training range. Preserve the
   categorical representation needed for the later resampling decision; scaling
   does not resolve fractional categorical values from ordinary SMOTE.

6. **Splits:** use stratified 80/10/10 training/validation/testing splits with
   `seeds.split: 42`. Persist each retained source record's split membership and
   share it across models. Do not search seeds for favorable performance.

Preparation order: audit/validate source and record identity; remove exact
original duplicates; create shared stratified splits; fit numerical imputation
and scaling using training records only; apply the same feature calculations
and fitted transformations to all splits; save audits, membership, feature order,
and preprocessing provenance. Deterministic operations may run before splitting
only when they do not learn from the full dataset.

## Implementation and pending work

Preparation, paired training/inference, evaluation, three-stage selection, and
SHAP are implemented. The canonical protocol supplies candidate ranges and
selection rules. Final selected values come only from completed stages.
The researcher application workflow is implemented. Receipt mapping/OCR and its
approved temporary-storage lifecycle remain to be implemented; see
[the implementation plan](BACKEND_IMPLEMENTATION_PLAN.md).

## Phase 2 implementation details

- Split generation uses scikit-learn stratified `train_test_split` twice: 80/20,
  then 50/50 of the held-out portion, both with seed 42. Integer rounding applies;
  both classes must be present in every resulting split.
- Exact-duplicate comparison uses complete canonical typed records across all
  11 original columns, before imputation/feature engineering. Numeric formatting
  differences are normalized through parsing; signed zeros compare equally.
  A temporary SQLite index compares full records without hash-collision risk.
- A missing original amount is retained in source/audit data and replaced by the
  training amount median only when features are generated. The zero indicator
  then describes that completed amount. Missing transaction type uses all-zero
  type indicators. Missing excluded balances/flags are audited and retained,
  without influencing the feature matrix.
- Min-Max scaling is fitted incrementally on training batches using scikit-learn.
  A JSON state stores exact coefficients, extrema, median, and feature order;
  application uses those values without refitting or clipping. Constant training
  ranges use scikit-learn's constant-feature behavior.
- Saved source rows preserve the categorical representation and provenance.
  Ordinary SMOTE was subsequently selected; preparation itself does not resample.
- Bundles retain source/configuration/split/code fingerprints and dependency
  versions. Labels reside in the split manifest, separately from feature files.
  The row-identity column is excluded by the shared feature loader.
- Failed or interrupted preparation is not a usable experiment bundle. A unique
  output directory and explicit completion status prevent silent overwrites and
  reuse of partial results.

## Completed PaySim preparation

Run: `paysim_phase2_20260918`, Python 3.12.5 on Windows. The full audit verified
6,362,620 source records with no missing fields, exact duplicates, or invalid
records. No imputation replacements were needed on this file. All 16 zero-amount
transactions were retained, and the original CSV fingerprint remains unchanged.

| Split | Records | Legitimate | Fraud |
| --- | ---: | ---: | ---: |
| Training | 5,090,096 | 5,083,526 | 6,570 |
| Validation | 636,262 | 635,440 | 822 |
| Testing | 636,262 | 635,441 | 821 |

Artifacts are local under `backend/data/prepared/paysim_phase2_20260918/`.
The separate verification report is under
`backend/reports/paysim_phase2_20260918/verification.json`.
All feature rows were checked for valid values and identity/label alignment;
all saved file hashes matched, and saved preprocessing exactly reproduced 100
records per split. The full 106-test suite passed. Reproducibility and protection
against held-out data influencing preprocessing were also tested synthetically.

These were preparation findings, not model-performance results. At the end of
Phase 2, RF/SMOTE training had not run; its later completion is recorded below.
SHAP and evaluation remain pending. The manuscript itself remains unedited.

## Training and sequential selection

Ordinary SMOTE applies only to original training features with k=5 and seed 42.
Benchmark RF uses the unchanged training distribution. Both models use identical
forest settings and model seed 42. The protocol determines the ratio and forest
candidates; the generic trainer's defaults are not finalized settings.

Preserve synthetic fractional encoded indicators and time features; do not round,
truncate, select a category by argmax, or filter artificial combinations. Synthetic
rows are numerical training vectors, not realistic receipts. Audit class counts,
new fraud rows, fractional indicators, and mixed type indicators for every fit.

Scores are mean tree fraud-class probabilities. Prediction is fraud when the
unrounded score is at least the common threshold. The 0.50 checkpoint in Stages
1 and 2 does not select a threshold. Communication bands are independent of this
classification boundary and are not calibrated real-world fraud probabilities.

See [the protocol](THREE_STAGE_VALIDATION.md) for all eight ratios, 12 forests,
objectives, exact tie rules, freezing, verified reuse, and recovery. Training
loads only training features; selection evaluates validation features only.
No combined training/validation refit is performed. Official held-out evaluation
requires a completed three-stage selection and separate authorization.

## Evaluation and explanations

- PR-AUC uses Average Precision, not trapezoidal integration. Retain continuous
  scores as well as predictions. F1 uses counts directly; undefined values carry
  explicit reasons. Accuracy is supplementary.
- Signed comparison is 100*(S-B)/((S+B)/2), S=RF-SMOTE and B=benchmark RF. Both
  zero means unavailable; if either MCC is negative, use S-B in coefficient units.
- Continuity-corrected McNemar is primary at alpha=.05. Add an exact two-sided
  binomial supplement for 1-24 discordances. Zero discordances use p=1 by convention
  and no statistic. It concerns paired error rates, not significance of every metric.
- Interventional TreeSHAP explains fraud probability using a shared uniform sample
  of 200 original-training records without replacement, seed 42, without balancing.
  A separate uniform 1,000-record evaluation sample supports global summaries;
  full-report explanation is an explicit job. Report actual coverage.
- SHAP reconstruction tolerance is 1e-6; positive-contributor tolerance is 1e-9.
  The prediction-equivalent representation adapter handles float32 thresholds and
  large trees without changing saved predictions or the explanation target.

See [implementation details](PHASE4_IMPLEMENTATION.md),
[manuscript guidance](THESIS_DOCUMENT_CHANGES.md), and
[current execution evidence](VALIDATION_RUN_STATUS.md).

## Application decisions

Top positive contributor and deterministic prose apply to either prediction class;
expanded details use a waterfall only. The numerical table is suggestion-only.
On 2026-09-30, the approved experimental receipt scope expanded to selected GCash app
workflows across all five PaySim categories: Express Send/TRANSFER, over-the-counter cash-in/CASH_IN,
over-the-counter cash-out/CASH_OUT, wallet-funded merchant QR or Pay Online/PAYMENT, and bank
transfer/DEBIT. This supersedes the earlier person-to-person-only restriction.
These are semantic demonstration mappings, not verified cross-domain equivalence.
Names, reference IDs, balances, and ground truth are not predictor inputs.
Approved mapping assumptions are Asia/Manila hour, Monday=0 weekday, and numeric
PHP principal excluding fees, with saved transformations reused once. Local OCR,
editable completion after image upload, explicit confirmation, optional string
reference, and temporary application storage until Clear or backend shutdown/restart
are settled requirements awaiting implementation. For personal-wallet origins,
approved merchant flags (origin/destination) are 0/0 for TRANSFER, 0/1 for PAYMENT,
and 0/0 for DEBIT, retaining the bank role separately. Cash-agent mappings remain
pending. The [local OCR baseline](RECEIPT_OCR_BENCHMARK.md) provisionally selects
RapidOCR; accept one PNG/JPEG up to 10 MiB / 20 million pixels, release
verified workflows first, and provide an individual ZIP without images or evaluation
metrics. Layout release checks and confirmed-role validation remain pending; the
OCR comparison is separate from fraud evaluation. The
[confirmed-receipt adapter](RECEIPT_MAPPING.md) now implements the approved mapping
and shared scaling with synthetic parity tests; receipt HTTP integration remains.
No feature semantics,
model artifacts, or research settings are changed by this scope decision.
See [receipt decisions and sample collection](RECEIPT_WORKFLOW.md).
See [mock-up decisions](MOCKUP_EVALUATION_DECISIONS.md) and
[help content](UI_HELP_CONTENT.md).
