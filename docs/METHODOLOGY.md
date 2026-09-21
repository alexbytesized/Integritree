# Methodology decision record

Last updated: 2026-09-22

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

## Unresolved decisions by dependent phase

| Phase | Decisions required |
| --- | --- |
| 2: preparation | Complete: approved policies implemented, tested, and verified on the supplied source. |
| 3: training | Complete for the fixed baseline. Seeds, neighbors and the bounded later validation procedure are settled below; no official test evaluation or validation search has run. |
| 4: evaluation and SHAP | Select PR-AUC integration and small-discordance McNemar handling. Implement approved signed-comparison/undefined/zero-discordance presentation. Finalize SHAP output/background/coverage and selected-run artifact support. |
| 5: application | Final prediction/research routes, batch handling, artifact compatibility, persistence. |
| 6: receipts | Supported GCash layouts, OCR, receipt-to-feature mapping, confirmation/fallback, and retention. |

`experiment.yaml` leaves unresolved parameters null. The configuration validator
can report whether the currently represented fields are filled; it cannot establish
scientific adequacy or completeness of the method. Extend its schema when later
decisions require additional controls. For example, the duplicate policy must also
prevent duplicate records from crossing splits if records are retained.

The feature, scaling, time, cleaning, and split choices above are approved.
Preparation and baseline training, including ordinary SMOTE at 1:1, are implemented
and verified. The Phase 3 section below records the settled training configuration
and approved later validation procedure. Evaluation and SHAP remain future work.

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

## Phase 3 implementation and approved tuning protocol

Phase 3 and full PaySim baseline training are complete and verified (2026-09-20).
Run: paysim_phase3_baseline_20260920. Both models were trained with all approved
records/settings and saved under backend/artifacts/paysim_phase3_baseline_20260920/.
The first attempt stopped before training because memory was insufficient.
After the researchers closed unused applications, the retry passed with 4.94 GiB
available. Recorded training duration was 9,594.625 seconds (about 2 h 40 min).
The raw CSV and all prepared files retain their original fingerprints.
Validation tuning, official test evaluation, and SHAP have not run.

### Implemented baseline

- Ordinary SMOTE, 1:1 minority:majority ratio, k_neighbors=5, SMOTE seed=42.
  The previously proposed neighbor count and seeds are carried forward under the
  instruction to implement Phase 3; they are explicit configuration values.
- Both RF models: 100 trees, maximum depth 20, minimum split size 2, minimum leaf
  size 1, sqrt feature selection, bootstrap=True, class_weight=None, criterion=gini,
  and model seed=42. Tree worker count is an execution option, shared by both.
- Convert the encoded training matrix to float64 before SMOTE. Preserve synthetic
  fractional indicators and time features; do not round, truncate, assign a category
  by argmax, or filter artificial combinations. RF internally uses its supported
  floating-point precision; no categorical rounding is introduced. Synthetic
  targets remain fraud (1).
- Audit original/synthetic counts, balanced counts, fractional indicators, and
  rows with multiple positive type indicators. Mixed synthetic type indicators
  have no single valid real transaction type. Numeric validity does not guarantee
  realistic transactions or improved performance.
- Shared score: class-1 predict_proba, the mean of tree fraud probabilities.
  Shared threshold: 0.50, including exact ties as fraud. These scores have not
  been demonstrated to be calibrated real-world fraud probabilities.
- Training loads only the original training split. No validation/test feature
  records are loaded by the training command, and neither split is resampled.
- Save both models, preprocessing, configuration, preparation provenance,
  feature order, software versions, implementation hashes, audit, and reload
  verification in a new run bundle. Never overwrite existing runs.
  Reject incomplete/failed bundles, corruption, schema/version mismatches.
  Only trusted local joblib files may be loaded.
- Single-record and batch inference use the same saved transformations/scoring.
  Labels and identity columns never enter the classifiers.
- Phase 3 verification at implementation: 123 tests passed, with two existing third-party deprecation warnings.
  Checks include repeatability, preserved original rows, fractional SMOTE types,
  training-only split loading, unchanged held-out fixtures, class-column lookup,
  exact ties, reload parity, paired identities, and individual/batch parity.

Confirmed full-run counts: 5,090,096 original training rows, 5,076,956 synthetic
fraud rows, and 10,167,052 RF-SMOTE training rows with 5,083,526 in each class.
The audit found zero fractional indicator values and zero rows with multiple
positive type indicators in this particular run. This does not remove the general
limitation of ordinary SMOTE on categorical features or change the preserve-fractions
policy; continuous engineered features can still take interpolated values.

Full-run verification passed: both saved models have 100 trees/max_depth=20 and
matching parameters; both serialization checks had maximum absolute score
difference 0.0 on 128 training rows. Saved preprocessing and raw-versus-prepared
prediction replay agreed on 128 training rows; individual-versus-batch predictions
agreed on 16 rows. All prepared-file and raw-source hash checks passed.
These are integrity/consistency checks, not performance evaluation. The report is
backend/reports/paysim_phase3_baseline_20260920/verification.json.

### Approved later validation search; not executed by Phase 3

The user accepted this protocol after the explanatory discussion:

1. Train both variants for each (trees, depth) candidate:
   (100,10), (100,20), (200,10), (200,20). Keep other RF settings matched.
2. At common cutoff 0.50, choose the shared candidate with highest mean validation
   F1 across RF and RF-SMOTE. Exact ties prefer shallower trees, then fewer trees.
3. With that shared RF configuration, test common cutoffs 0.05, 0.10, ..., 0.95.
   Select highest mean validation F1 across both; exact ties prefer nearest
   to 0.50, then the higher cutoff. Do not choose separate model cutoffs.
4. Implement Phase 4 metrics first, then run this bounded validation search,
   freeze shared settings/cutoff, and only then perform final test evaluation.
   Do not repeatedly expand the search based on observed validation results.

A limited search does not guarantee no overfitting or global optimality.
The shared cutoff need not individually maximize each model's F1.
The fixed-baseline trainer does not execute selection, tuning, or final testing.
Implement a recorded selection artifact in Phase 4.

### Still unresolved for Phase 4 and later

- Equation 10 gives an integral PR-AUC definition without selecting AP versus
  trapezoidal integration. The earlier AP choice is reopened; pr_auc_method is
  null. PR-AUC is not required by baseline training and belongs with other
  Phase 4 metrics, using scores rather than a single confusion matrix.
- McNemar calculation and small-discordance handling. See the later presentation
  decisions below for approved zero-discordance and undefined-value display.
- SHAP output space, background, size, and coverage.
- Receipt mapping, OCR, retention, and application workflow decisions.

See [THESIS_DOCUMENT_CHANGES.md](THESIS_DOCUMENT_CHANGES.md) for the manuscript
alignment checklist. No manuscript edits have been made.

### UI-review decisions approved 2026-09-22; pending implementation

- Use signed symmetric percentage difference: 100*(S-B)/((S+B)/2), with
  S=RF-SMOTE and B=benchmark RF. This replaces the earlier absolute convention
  for future evaluation presentation. Positive means RF-SMOTE is higher; negative
  means benchmark RF is higher. It is not a significance test or baseline-relative
  percentage change.
- Apply that percentage to nonnegative scores with a positive mean. Both zero
  gives N/A (zero denominator). If either MCC is negative, compare S-B in coefficient
  units. Undefined metrics/comparisons show N/A with a reason, not an invented zero.
- With no McNemar discordant pairs, display p=1 by convention, statistic N/A,
  and No discordant pairs. The small-discordance method still needs finalization.
- PR-AUC integration remains unselected. General approval of help content does
  not choose between AP and trapezoidal integration.
- Active configuration now selects signed_over_mean; the schema accepts it and
  legacy absolute_over_mean (retaining the legacy default for older snapshots).
  Phase 4 calculation remains pending. Preserve immutable saved-run configurations
  and record the evaluation policy separately, exporting the actual method used.

### Completed mock-up review and backend audit

On 2026-09-22 the expanded SHAP view was narrowed to a waterfall graph only.
The numerical table remains a suggestion. Top positive contributor (including for
legitimate predictions) and deterministic per-model prose remain approved. Preserve
the numerical attribution object internally for chart generation and additivity;
the UI choice does not change the underlying explanation method. Background,
runtime coverage and exact explainer configuration still need finalization.

Approved score communication bands use unrounded 100*p: Minimal [0,20), Low [20,40),
Moderate [40,60), High [60,80), Critical [80,100]. They are independent of the saved
classification threshold. The ordinary user's receipt workflow remains experimental
GCash person-to-person TRANSFER only, under explicit timing/amount assumptions.
No real names, reference IDs, or excluded balances become model features.

The code audit found Phases 1-3 reusable for this scope. The current raw-record
preprocessor is not a derived-input receipt adapter; add one with unchanged saved
scaling in Phase 6. The current artifact loader enforces baseline settings; add
explicit validation-selected run support in Phase 4. These are future extensions,
not reasons to overwrite the completed baseline or redo preparation.

Audit verification: 125 tests passed, including signed/legacy configuration and
synthetic saved-inference compatibility. No official test evaluation, retraining,
or manuscript editing occurred. See [the audit](BACKEND_ALIGNMENT_AUDIT.md).

See [the mock-up decision record](MOCKUP_EVALUATION_DECISIONS.md) and
[approved help content](UI_HELP_CONTENT.md) for frontend/export requirements.
