# Methodology decision record

Last updated: 2026-09-19

This is an implementation decision record, not a revised Chapter 3.
Read [THESIS_CONTEXT.md](THESIS_CONTEXT.md) for the manuscript requirements and
researchers' confirmations. Recognizing an option in a configuration schema does
not approve using it in the experiment.

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
the research dataset to those types. Sixteen records have zero amounts; keep valid
unusual transactions. There is no calendar date/timezone anchor in the CSV, so
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
The internal score field is bounded to 0..1; this does not settle how the score is
computed, whether it is calibrated, or its frontend display scale. SHAP contracts
name the output space but do not yet compute explanations or verify additivity.

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

## Unresolved decisions by dependent phase

| Phase | Decisions required |
| --- | --- |
| 2: preparation | Complete: approved policies implemented, tested, and verified on the supplied source. |
| 3: training | Ordinary SMOTE, 1:1 balance, starting RF settings and probability scoring approved below. Matching RF settings, validation F1 for threshold selection, and AP are also selected. Neighbor count, model/SMOTE seeds, shared RF tuning and threshold search/reporting rules remain open. |
| 4: evaluation and SHAP | PR-AUC integration; McNemar variant and zero/few-discordance behavior; undefined metric/percentage handling; SHAP output, background and coverage. |
| 5: application | Final prediction/research routes, batch handling, artifact compatibility, persistence. |
| 6: receipts | Supported GCash layouts, OCR, receipt-to-feature mapping, confirmation/fallback, and retention. |

`experiment.yaml` leaves unresolved parameters null. The configuration validator
can report whether the currently represented fields are filled; it cannot establish
scientific adequacy or completeness of the method. Extend its schema when later
decisions require additional controls. For example, the duplicate policy must also
prevent duplicate records from crossing splits if records are retained.

The feature, scaling, time, cleaning, and split choices above are approved.
Ordinary SMOTE and 1:1 balancing are now approved; the remaining training
choices listed below and later-phase decisions remain unresolved.
The preparation command is implemented and tested; model training remains
unimplemented. No SMOTE has been applied. See the implementation plan for
full-dataset execution and verification status.

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

These are preparation findings, not model-performance results. RF, SMOTE, SHAP,
and evaluation have not run. The manuscript itself remains unedited.

## Phase 3 discussion: decisions confirmed on 2026-09-19

The researchers cannot replace ordinary SMOTE with SMOTENC. They approved 1:1
minority-to-majority training balance and the proposed starting RF settings:
100 trees, maximum depth 20, minimum split size 2, minimum leaf size 1, square-root
feature selection, bootstrap enabled, and no class weighting. These are starting
settings, not a finalized tuning protocol or claims of optimality.

They approved mean-tree fraud probability scoring, an initial 0.50 threshold,
and fraud classification on an exact threshold tie, while asking to allow
threshold exploration. The researchers subsequently chose highest validation-set F1 as the threshold
selection objective and confirmed 0.50 as the common baseline. The candidate
search/tie-breaking rules and whether selected thresholds are shared or
model-specific remain unresolved.

Float-handling implementation policy for the retained ordinary SMOTE method:
convert the full encoded training feature matrix to floating point before
resampling, retain fractional synthetic indicator/time-feature values, and never
cast those synthetic features back to integer indicator dtypes or round them to
categories. Real prepared records retain their original 0/1 indicator values;
labels remain integer 0/1, with generated minority labels equal to 1. Synthetic
rows are training feature vectors, not authentic transaction records or receipts.
Record synthetic counts and audit fractional/semantically inconsistent values;
this audit does not silently repair, filter, or round synthetic samples.

The researchers explicitly confirmed matching RF settings so oversampling is
the only intended training difference, and validation-only selection before final
test evaluation. Fixed versus systematically tuned shared RF settings remains
open. The recommended primary common-cutoff comparison plus separately reported
model-specific tuned thresholds has been explained but not yet approved. Parameter changes require a new trained run; threshold
changes can reuse prediction scores but require a recorded decision policy.

Still awaiting decisions: fixed versus tuned shared RF settings, threshold
selection/reporting protocol, SMOTE k_neighbors (5 proposed), and model/SMOTE
seeds (42 proposed). AP has subsequently been selected as the precision-recall
summary; report it explicitly as Average Precision, distinct from trapezoidal
PR-AUC. It is computed from scores, not one thresholded confusion matrix.
No Phase 3 training or oversampling has been executed.

Clarification for the researchers: one-hot encoding maps transaction type into
five 0/1 indicators. Changing their storage from integer 0/1 to floating-point
0.0/1.0 changes no values or meanings. Fractional intermediate values may be
created later by SMOTE interpolation; they are not introduced by that numeric
conversion. The researchers requested this explanation, and no oversampling or
training has yet been performed.
