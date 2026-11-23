# Reproducing the 100-tree time-sensitivity experiment

Prepared: November 23, 2026.

**Status: documentation only.** This guide does not authorize a branch switch,
reset, deletion, implementation, training, evaluation, or deployment. Wait for
the user to confirm that the clean branch is ready before implementing or running
the experiment described below. Creating this document does not create its
proposed runner or output directories.

Keep a copy of this document somewhere available after switching branches. Do
not assume that an untracked file will be preserved by every reset/cleanup workflow.

## 1. Objective and agreed scope

Investigate whether reducing Random Forest tree depth or increasing minimum leaf
size reduces reliance on `hour_of_day` and `day_of_week`, while measuring the
effect on fraud detection. Compare ordinary RF with RF trained on SMOTE-augmented
training data (RF-SMOTE) under matched settings.

Reproduce **only the 100-tree experiment**, using:

- The original prepared training split and its already-fitted preprocessing.
- The original saved model pairs that existed before the time-sensitivity work.
- `Experiment-Dataset.csv` as the **only validation dataset for this reproduction**.
- Six paired configurations, fixed-43% evaluation, independently selected shared
  cutoffs for each pair, SHAP diagnostics, and hour/day sensitivity sweeps.

Do not reproduce the 200-tree experiment, the earlier original-validation run,
or the separate three-stage ratio/forest selection experiment. Do not repartition
the supplied CSV, use it for training, or evaluate the official held-out test set.
The final application model and cutoff remain unchanged.

This is a direct-to-CSV reconstruction of the methodology. Historically, the
missing 100-tree models were first trained in another run and then reused for CSV
evaluation. The clean reconstruction trains those missing models directly from
the same original training split; it does not need that intervening run.

## 2. Starting state and original inputs to preserve

Paths below are relative to the repository root unless explicitly absolute.
They identify the known source artifacts; verify their contents on the clean
branch rather than assuming that matching folder names establish compatibility.

| Required input | Known location | Use |
|---|---|---|
| Prepared training bundle | `backend/data/prepared/paysim_phase2_20260918/` | Training features, aligned labels, preprocessing state, and metadata |
| Original model-selection artifacts | `backend/artifacts/paysim_three_stage_20260928_172539/` | Original saved model pairs and provenance |
| Supplied validation CSV | `C:/Users/Carpicorn/Downloads/Experiment-Dataset.csv` | Evaluation labels and already-engineered predictors |
| Backend source | `backend/src/integritree/` | Existing feature, training, inference, metric, and artifact utilities |
| Python environment and dependency definitions | `backend/.venv/`, `backend/pyproject.toml`, `backend/requirements-dev.lock` | Compatible execution environment |

The prepared files actually needed for training are `metadata.json`,
`configuration.json`, `preprocessing.json`, `train_features.parquet`, and
`split_manifest.parquet`. Filter the split manifest to **train** when obtaining
labels. Original validation/test feature files are not inputs to this workflow.

Retain complete saved model bundles, not just the `.joblib` files: their loaders
also require configuration, preprocessing, training audit, provenance, file
fingerprints, and reload-verification evidence.

### Original reusable model pairs

All three pairs below live under the original artifacts directory's `candidates/`:

| Depth / leaf | Candidate directory | Recorded training directory |
|---|---|---|
| 10 / 1 | `ratio_0.01_trees_100_depth_10_leaf_1` | `train_20260928T092543_93f9a4e8` |
| 10 / 10 | `ratio_0.01_trees_100_depth_10_leaf_10` | `train_20260928T143535_5610b55d` |
| 10 / 50 | `ratio_0.01_trees_100_depth_10_leaf_50` | `train_20260928T144913_39813db2` |

Each directory contains RF and RF-SMOTE. Verify 100 trees, the intended depth and
leaf size, seeds, original prepared-data fingerprint, and identical preprocessing
before reuse. Raw training bundles may record a 50% scoring threshold; this is
not a different fitted forest. Apply this experiment's 43% and selected-cutoff
policies during evaluation without editing the saved bundle.

If an original pair is missing, ambiguous, or incompatible, report that finding
before proceeding. Do not silently substitute a 200-tree model or change the
planned fit count. Models produced by previous time-sensitivity runs must not
be required inputs to the fresh experiment.

### Recorded data identity

| Data | Rows | Fraud | Legitimate |
|---|---:|---:|---:|
| Original training split | 5,090,096 | 6,570 | 5,083,526 |
| Supplied validation CSV | 636,262 | 822 | 635,440 |

Recorded SHA-256 for the validation CSV:

```text
4a883825dfe58a8f9e445131e06c4251ce2c7fa9e8a29e1490f667b27d961b1d
```

Record the actual fingerprint when reproducing the run. Different bytes or row
order can change predictions or diagnostic samples; do not silently describe a
different file as an exact reproduction. These checks establish file identity,
not dataset independence. Dataset provenance and training overlap were not
investigated for the CSV experiment and are not asserted by this guide.

### Recorded environment

Use Python 3.12 and preserve the compatible backend environment where available.
The historical run recorded NumPy 2.5.3, pandas 2.3.3, PyArrow 23.0.1,
scikit-learn 1.9.1, imbalanced-learn 0.14.2, joblib 1.6.0, SHAP 0.52.0,
and integritree-backend 0.1.0. Preserve the branch's complete lock/configuration
files as well; this list is not a complete replacement for a dependency lock.
Saved artifact loaders check package compatibility. Do not bypass those checks
or silently upgrade the environment to make an incompatible artifact load.

## 3. Experimental design

All forests use **100 trees**. Train/evaluate RF and RF-SMOTE for each row:

| Configuration | `max_depth` | `min_samples_leaf` | Planned source |
|---|---:|---:|---|
| Baseline | 10 | 1 | Reuse original pair |
| Reduced depth | 8 | 1 | Train two new models |
| Further reduced depth | 6 | 1 | Train two new models |
| Increased minimum leaf size | 10 | 10 | Reuse original pair |
| Further increased minimum leaf size | 10 | 50 | Reuse original pair |
| Combined restriction | 8 | 10 | Train two new models |

Expected total: **six new models plus six reused models = 12 evaluated models**.

Hold these settings fixed:

| Setting | Value |
|---|---|
| `n_estimators` | 100 |
| `criterion` | `gini` |
| `max_features` | `sqrt` |
| `min_samples_split` | 2 |
| `bootstrap` | `True` |
| `class_weight` | `None` |
| Model seed / SMOTE seed | 42 / 42 |
| Training workers | 1 |
| Features and preprocessing | Original 11-feature schema and training-fitted state |
| SMOTE method | Ordinary SMOTE on the encoded/scaled training features |
| SMOTE neighbors | 5 |
| SMOTE sampling ratio | 0.01, final fraud-to-legitimate ratio approximately 1:100 |

Verify the full saved forest parameter set and inherit other defaults consistently.
Change only depth and minimum leaf size between the six configurations.

Lower `max_depth` limits successive splits along a tree path. Higher
`min_samples_leaf` requires more samples in each leaf, counting both classes and,
for RF-SMOTE, synthetic samples. Neither setting specifically penalizes time.
Less complex trees can reduce time effects, remove useful signal, or change
relative feature shares without reducing absolute time effects.

## 4. Load data without changing its meaning

Use this exact model-input order:

```text
hour_of_day
day_of_week
type_CASH_IN
type_CASH_OUT
type_DEBIT
type_PAYMENT
type_TRANSFER
log_amount
is_zero_amount
is_merchant_origin
is_merchant_dest
```

The CSV must contain these 11 columns plus `isFraud`, with no duplicate column
names. Read numeric values with round-trip float precision. Reject missing or
non-finite feature values, non-binary labels, an empty file, or missing label
classes. Arrange predictors in the saved feature order and remove `isFraud`
from the model input matrix.

**Do not fit preprocessing, engineer raw features, scale again, impute, drop
duplicate CSV rows, or change labels.** The supplied CSV already contains
engineered/scaled features. Preserve its row order and assign one-based data-row
positions, excluding the header, as local identities. Namespace those positions
by the CSV fingerprint; they are not original PaySim transaction IDs.

For orientation, the original feature convention is:

```text
raw hour = (step - 1) % 24
raw day  = ((step - 1) // 24) % 7
raw log amount = log1p(amount)
```

The recorded training scaler maps hour 0..23 and day 0..6 to 0..1. Its amount
scale is 0.054519295105796506 and its offsets are zero. Read the saved state
rather than hardcoding these numbers into inference. These definitions explain
the data; they are not instructions to transform the validation CSV again.

## 5. Training procedure

1. Load and verify original artifacts and the prepared bundle. Check available
   memory before materializing training/resampled arrays; the existing working-
   array estimate is about 3.0 GiB before extra tree/runtime overhead.
2. Load only the original training features and the matching training labels.
   Check their source-row alignment and preserve row order.
3. Build one ordinary-SMOTE sample from that training matrix and its labels,
   using ratio 0.01, five neighbors, and seed 42. The validation CSV must never
   enter this operation.
4. Preserve the original training prefix and save the synthetic tail plus an
   audit and fingerprint. Expected augmentation: **44,265 synthetic fraud rows**,
   giving **5,134,361 total rows**, including 50,835 fraud rows. The small departure
   from an exact 0.01 ratio comes from integer sample counts.
5. Train the new 8/1, 6/1, and 8/10 RF models on original training data, and their
   RF-SMOTE counterparts on the same shared augmented matrix. Train sequentially.
6. Save each model in the new isolated output folder, reload it, verify its
   parameters, and compare predictions before/after reload (`atol=1e-12`,
   `rtol=0`). Only then mark its checkpoint complete.

Keep ordinary SMOTE's generated numeric values. Do not introduce categorical
rounding, SMOTENC, a different ratio, or separate random resamples per candidate.
Those would change the experiment.

## 6. Prediction, metrics, and shared-cutoff selection

Score every CSV row with every model, in a consistent row order, preferably in
25,000-row batches. Use the probability of class 1, not rounded display scores.
Save scores once and derive all cutoff decisions from those saved values.

Evaluate two policies:

1. **Fixed:** 43% for every model, supporting comparisons at an unchanged cutoff.
2. **Selected:** one shared cutoff per RF/RF-SMOTE pair, chosen from exactly
   1%, 2%, ..., 100% to maximize that pair's average F1.

```text
predicted fraud = score >= cutoff
precision = TP / (TP + FP)
recall = TP / (TP + FN)
F1 = 2*TP / (2*TP + FP + FN)
pair average F1 = (RF F1 + RF-SMOTE F1) / 2
```

Compare unrounded F1 objectives using exact rational arithmetic or integer
cross-products. For exact ties, prefer the cutoff closest to 50%; if still tied,
prefer the higher cutoff. Independently verify confusion counts at all 100
cutoffs for all six pairs. The selected average F1 cannot be lower than the
fixed-43% average because 43% is included in the search grid.

A shared optimum is not necessarily the optimum for either model individually.
It is also not a search over every possible real-valued threshold. Do not
replace the protocol with separate per-model cutoffs or an AP-based forest search.

Report precision, recall, F1, pair average F1, **PR-AUC calculated as average
precision (AP)**, TP (fraud detected), FP (false alarms), TN, and FN. Preserve
undefined-metric reasons rather than inventing values when a denominator is zero.
AP uses the full score ranking and is unchanged by the chosen cutoff.

## 7. SHAP protocol and time-contribution calculations

Explain fraud probability using interventional Tree SHAP. Use the same background
and the same sampled validation records for all 12 models.

### Sampling

For each pool below, independently initialize `numpy.random.default_rng(42)`,
sample without replacement, and sort the selected positions. Do not carry one
generator's state from one pool to the next.

```python
def sample(pool, count):
    return np.sort(
        np.random.default_rng(42).choice(
            pool, size=min(count, len(pool)), replace=False
        )
    )
```

| Sample | Pool | Requested size |
|---|---|---:|
| SHAP background | Original training rows, before SMOTE | 200 |
| Uniform validation cohort | All supplied CSV positions | 200 |
| Actual-fraud cohort | CSV positions with `isFraud == 1` | 100 |
| Baseline-RF-flagged cohort | Baseline 10/1 RF scores >= 0.43 | 50 |
| Baseline-RF-SMOTE-flagged cohort | Baseline 10/1 RF-SMOTE scores >= 0.43 | 50 |
| Time-sweep cohort | All supplied CSV positions | 50 |

Freeze cohort membership using the baseline before evaluating alternatives.
The four SHAP cohorts overlap. Explain their unique union once per model and
then summarize each cohort separately. The recorded CSV run had **383 unique
SHAP records**, not 400; do not force that count if inputs differ. Keep empty
cohorts explicit with undefined summaries, and use all available rows if a pool
has fewer records than requested.

Main comparison tables use the **uniform validation cohort**. Fraud and flagged
cohorts describe deliberately selected subsets and must be labeled accordingly.
Changing a selected cutoff does not change SHAP values or reselect these cohorts.

### Numerical compatibility and reconstruction

Use the backend's prediction-equivalent `forest_representation` adapter when
available and verified on the clean branch. Do not assume that a native SHAP
constructor supports every saved forest representation correctly.

Set `model_output="probability"`, `feature_perturbation="interventional"`, and an
independent masker retaining all 200 background rows. Match sklearn's float32
prediction inputs for both the explained values and the SHAP background. Preserve
the original stored feature values; this is an explanation/input-precision step,
not refitting or rescaling.

The float32-background handling is a numerical correction learned during later
verification. The historical 100-tree SHAP values below were produced by the
earlier implementation; minor numerical differences are possible when applying
the correction. Record the implementation used instead of forcing exact SHAP
agreement or loosening checks.

Check adapter predictions against the original forest (`atol=1e-12`, `rtol=0`).
For every explained record, require:

```text
abs(base fraud probability + sum(all 11 SHAP values) - model fraud probability)
    <= 1e-6
```

Extract the class-1 SHAP values, and save each record's feature contributions,
base probability, model score, and CSV row identity.

### Time summaries

Let `phi_hour(i)` and `phi_day(i)` be the time-feature contributions for record
`i`, and `phi_j(i)` the contribution for feature `j`. For a cohort of `n` records:

```text
Mean absolute time contribution (score percentage points):
    100 * sum_i(abs(phi_hour(i)) + abs(phi_day(i))) / n

Time share of total absolute SHAP (fraction):
    sum_i(abs(phi_hour(i)) + abs(phi_day(i)))
    / sum_i(sum_j(abs(phi_j(i))))
```

Multiply the second value by 100 only when displaying it as a percentage.
Return an undefined share if the total absolute contribution is zero. Compute
the ratio of cohort totals, not the mean of per-record ratios. Take absolute
values before adding hour and day, so opposing contributions do not cancel.

For example, hour `+0.03` and day `-0.02` contribute **5 percentage points** by
the absolute definition, although their signed net contribution is +1 point.
Time share is not time's percentage of the final fraud score.

For continuity with the earlier detailed diagnostics, also record
`100 * mean(abs(phi_hour + phi_day))` as net absolute time contribution. Group
absolute contributions into time, transaction type, amount, and entity features,
and report the fraction of rows where time is strictly larger than every other
group; ties do not count. Keep these supplementary measures distinct from the
two primary time columns.

## 8. Hour/day sensitivity sweep

For each of the 50 sampled CSV records, hold all non-time predictors fixed and
substitute all **24 hours x 7 days = 168 combinations**. Construct each time value
using the original fitted scaler's scale and offset for that feature.

For a reproducible grid, use hours 0..23 within each day 0..6. Score the 8,400
substituted rows per model and save the complete grid. For each original record,
calculate minimum score, maximum score, and:

```text
range in score percentage points = 100 * (maximum score - minimum score)
crosses cutoff t = minimum score < t AND maximum score >= t
```

Summarize mean/median range and the fraction crossing 43% or that pair's selected
cutoff. The sweep generates **100,800 scores across 12 models**.

These are diagnostic substitutions, potentially outside the observed data
distribution. They do not establish causal time effects, and the substituted
rows must not be used as labeled transactions for accuracy or F1 evaluation.

## 9. Reconstruct an isolated runner after the clean branch is confirmed

Suggested future folder: `time_sensitivity_100_reproduction/` at the repository
root. Keep all new code, configuration, tests, runtime caches, models, and results
inside it. Reference original artifacts read-only.

The old `backend/forest_regularization_experiment/evaluate_csv.py` is not a clean-
start entry point: it expects a completed earlier experiment, saved predictions,
and prior cutoff results. The 200-tree runner also imports earlier experiment
helpers. **Do not make either experiment folder or its outputs a prerequisite.**

On the clean branch, inspect these original backend interfaces before reuse:

| Module | Relevant capability |
|---|---|
| `integritree.ml.artifacts` | `load_bundle`: verified saved-model loading |
| `integritree.ml.preparation` | `load_prepared_split`: aligned training features, labels, identities |
| `integritree.ml.preprocessing` | `FittedPreprocessor`: read the original scaler state |
| `integritree.ml.features` | `FEATURE_COLUMNS`: authoritative feature order |
| `integritree.ml.inference` | `feature_matrix`, `fraud_scores` |
| `integritree.ml.training` | `forest_parameters`, `memory_preflight`, `resample_training` |
| `integritree.ml.evaluation` | Metrics and undefined-value policies |
| `integritree.ml.threshold_search` | Exact shared-F1 percent-grid selection, if present |
| `integritree.ml.shap_adapter` | Prediction-equivalent forest representation, if present |

These interfaces were present when this guide was prepared. A reset may expose
an older implementation. Verify behavior on that branch; if a helper is absent,
implement the required behavior inside the new experiment folder and test it.
Do not import helpers from deleted experiment directories or silently change
the original application's behavior to satisfy this workflow.

Recommended execution order for the future runner:

```text
read configuration and original source metadata
verify model/data/environment compatibility and CSV format
record input/configuration/code fingerprints and check memory
load original training split
select and save original-training SHAP background
create or restore one shared training-only SMOTE sample
train, save, reload-check, and checkpoint the six missing models
release full training/resampled arrays before explanation work
score the baseline on the supplied CSV and freeze diagnostic cohorts
for each of the six pairs:
    load verified models and score all CSV rows
    evaluate fixed 43% and select/verify the best shared cutoff
    calculate SHAP for the common union and summarize each cohort
    run the 50 x 168 time sweep
    save predictions, explanations, diagnostics, and a completion checkpoint
build comparison tables and offline report
independently verify results, source preservation, and output fingerprints
mark the run complete only after required checks pass
```

### Proposed command interface — not implemented by this document

The following commands are a specification for the future runner. **They are not
currently available merely because this Markdown file exists.** Implement and
test the runner after the user confirms the clean branch, then use fresh run IDs.
Run from the repository root in PowerShell:

```powershell
# Future preflight: no fitting or SHAP execution.
& .\backend\.venv\Scripts\python.exe -B .\time_sensitivity_100_reproduction\run.py

# Future full execution: explicitly choose an unused run ID.
& .\backend\.venv\Scripts\python.exe -B .\time_sensitivity_100_reproduction\run.py --execute --run-id repeat_100_csv_01

# Future resume of an unfinished run with compatible fingerprints.
& .\backend\.venv\Scripts\python.exe -B .\time_sensitivity_100_reproduction\run.py --execute --resume repeat_100_csv_01

# Future independent verification.
& .\backend\.venv\Scripts\python.exe -B .\time_sensitivity_100_reproduction\verify.py .\time_sensitivity_100_reproduction\outputs\repeat_100_csv_01
```

Expose input locations through configuration and `--config`; resolve relative
input paths against that configuration file. Output paths must remain inside the
new experiment folder. Preflight and execution must use distinct output folders.

## 10. Outputs, recovery, and acceptance checks

Produce at least:

- A manifest and plan recording source paths/fingerprints, dependency versions,
  full parameters, seeds, train/validation roles, completion state, and timings.
- Six newly trained model files, reload checks, and reusable model checkpoints.
- Shared SMOTE audit and synthetic-row checkpoint; original-training background
  and frozen validation/sensitivity cohort identities.
- Per-pair full scores, fixed and selected classifications, all 100 threshold
  choices, selected-cutoff decisions, SHAP vectors/base scores, and time sweeps.
- Machine-readable metrics and time summaries, differences from the baseline,
  plus Markdown and offline HTML comparison reports.
- A verification record and final output fingerprints.

Main tables must include configuration, model, cutoff, precision, recall, F1,
**pair average F1**, PR-AUC/AP, fraud detected, false alarms, absolute time
contribution, and time share. Separate fixed-43% and selected-cutoff policies.
Show other SHAP cohorts in separately labeled tables. Do not add an old-results
dependency merely to render a historical comparison.

Use atomic completion markers: a partially written model is not a completed fit.
Resume only when model, training data, CSV, configuration, implementation, and
environment fingerprints are compatible. Reuse verified completed fits; restart
unfinished fits. Preserve the same synthetic sample on resume. Verify completed
evaluation checkpoints before reuse; recalculate an unfinished pair evaluation.
Reject concurrent writers and existing completed output IDs. After an abrupt
process exit, confirm that its process is no longer active before removing only
that run's stale lock. Do not automate broad deletion or reset commands.

Record per-fit elapsed time and each execution/resume session separately. A hard
interruption can prevent a process from finalizing its total timer. Distinguish
successful-fit time, interrupted attempts, evaluation time, and offline gaps.
The historical CSV-only reevaluation took about four minutes because it reused
all models; it is not a training-time estimate for this fresh reconstruction.

Before declaring completion, verify:

- Six pairs / 12 models with the intended parameters; six new completed fits.
- CSV rows/duplicates/order preserved, labels excluded from predictors, training
  labels aligned, and no validation data in fitting, scaling, or SMOTE.
- Model reload and prediction equivalence; finite class-1 probabilities in [0, 1].
- Every confusion matrix sums to the validation row count and has 822 actual
  positives when using the recorded CSV; recomputed metrics match saved reports.
- All **600 cutoff choices**, score-equality behavior, exact tie rules, and
  unrounded pair averages; AP does not change with the cutoff policy.
- Reproducible cohort membership, 200 unsampled-down background rows, and SHAP
  score reconstruction within 1e-6 for every explained record.
- Time summary arithmetic and all 168 unique hour/day combinations per sampled
  record, with non-time inputs unchanged and crossing counts reproduced.
- Source fingerprints unchanged; complete output inventory; readable reports.

Use small synthetic tests for the CSV boundary, training-only SMOTE, exact ties,
checkpoint corruption/context rejection, interrupted-fit resume, evaluation
resume, and SHAP numerical compatibility before the full-data run. Synthetic
test metrics are software checks, not research findings.

## 11. Historical reference results on Experiment-Dataset.csv

These values are evidence from the completed 100-tree CSV evaluation, not
prescribed outcomes or a reason to tune until a match is obtained. They are
included directly so the guide remains useful if old output folders disappear.

### Metrics at each pair's selected shared cutoff

Pair average F1 is shown once per pair and was computed from unrounded values.

| Depth / leaf | Model | Cutoff | Precision | Recall | F1 | Pair average F1 | AP | TP | FP |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 / 1 (baseline) | RF | 46% | 89.00% | 44.28% | 0.5914 | 0.6189 | 0.6177 | 364 | 45 |
| 10 / 1 | RF-SMOTE | 46% | 65.70% | 63.63% | 0.6465 | — | 0.6061 | 523 | 273 |
| 8 / 1 | RF | 40% | 86.98% | 40.63% | 0.5539 | 0.5745 | 0.5962 | 334 | 50 |
| 8 / 1 | RF-SMOTE | 40% | 55.53% | 64.11% | 0.5951 | — | 0.5858 | 527 | 422 |
| 6 / 1 | RF | 41% | 92.76% | 24.94% | 0.3931 | 0.4894 | 0.5585 | 205 | 16 |
| 6 / 1 | RF-SMOTE | 41% | 60.70% | 56.57% | 0.5856 | — | 0.5538 | 465 | 301 |
| 10 / 10 | RF | 44% | 86.97% | 42.21% | 0.5684 | 0.6059 | 0.6074 | 347 | 52 |
| 10 / 10 | RF-SMOTE | 44% | 64.70% | 63.99% | 0.6434 | — | 0.5979 | 526 | 287 |
| 10 / 50 | RF | 45% | 86.44% | 37.23% | 0.5204 | 0.5810 | 0.5909 | 306 | 48 |
| 10 / 50 | RF-SMOTE | 45% | 64.19% | 64.11% | 0.6415 | — | 0.5955 | 527 | 294 |
| 8 / 10 | RF | 39% | 85.04% | 39.42% | 0.5387 | 0.5651 | 0.5929 | 324 | 57 |
| 8 / 10 | RF-SMOTE | 39% | 54.83% | 64.23% | 0.5916 | — | 0.5795 | 528 | 435 |

For an additional fixed-policy reference, the baseline at **43%** had RF F1
0.6076555024 (381 TP, 51 FP) and RF-SMOTE F1 0.6224310041 (530 TP, 351 FP).
This illustrates why the pair-optimal cutoff need not maximize RF's own F1.

### Historical uniform-cohort time summaries

| Depth / leaf | RF absolute time (pp) | RF time share | RF-SMOTE absolute time (pp) | RF-SMOTE time share |
|---|---:|---:|---:|---:|
| 10 / 1 | 0.2605 | 13.23% | 0.7350 | 19.57% |
| 8 / 1 | 0.2241 | 12.03% | 0.6563 | 18.96% |
| 6 / 1 | 0.1972 | 13.62% | 0.5880 | 19.53% |
| 10 / 10 | 0.2522 | 12.44% | 0.7641 | 21.04% |
| 10 / 50 | 0.2338 | 11.26% | 0.7479 | 19.54% |
| 8 / 10 | 0.2248 | 12.61% | 0.6584 | 18.90% |

Reference evidence, if retained, is in
`backend/forest_regularization_experiment/outputs/csv_validation_20261118/`
(`manifest.json`, `comparison.csv`, `time_shap.csv`, `pair_decisions.csv`, and
`verification.json`). These files are optional reading, **not runtime inputs**
for the clean reconstruction. The earlier run's plan and the original candidate
bundles supplied the model-origin information recorded above.

## 12. Interpretation and completion decisions

Compare each RF against baseline RF and each RF-SMOTE against baseline RF-SMOTE.
Use fixed-43% rows to examine behavior at the same cutoff, and selected-cutoff
rows to compare each pair after the agreed tuning procedure. Changing both the
model and its cutoff means a difference cannot be attributed to regularization
alone. Interpret absolute time contribution and relative time share separately.

Lower time influence alone is not success: a nearly constant low-score model
could have tiny time effects while missing most fraud. Examine recall, precision,
F1, AP, fraud detections, and false alarms together. More regularization can
underfit, and larger leaves do not guarantee less time dependence.

Results are exploratory validation findings because the same CSV is used for
cutoff selection and reporting. One seed and small SHAP samples do not establish
robustness or real-world fraud performance. The simulated hour/day meanings
must not be treated as evidence about real transaction calendars. Keep any
later final model selection, held-out testing, or deployment as separate work.

If a rerun differs, first compare inputs and row order, model origins and full
parameters, dependency versions, training order/SMOTE sample, positive-class
index, cutoff policy, SHAP background/dtype, and cohort membership. Report
differences honestly. Do not modify labels, force cutoffs to the historical
values, or weaken verification tolerances to recover an expected result.

## 13. Implementation brief to use after the branch is ready

Copy the following into a future session only after confirming the clean branch:

> The branch is now in its pre-time-sensitivity state. Reproduce the 100-tree
> experiment specified in TIME_SENSITIVITY_100_TREE_REPRODUCTION.md. First verify
> the available original prepared data, model bundles, and backend interfaces.
> Build a fresh isolated runner under time_sensitivity_100_reproduction/ without
> depending on earlier experiment code or outputs. Use only the original training
> split for new fits and Experiment-Dataset.csv for validation. Reuse compatible
> original 10/1, 10/10, and 10/50 pairs; train RF and RF-SMOTE for 8/1, 6/1, and
> 8/10 with 100 trees and the fixed protocol. Include fixed-43% and pair-optimal
> shared-cutoff results, pair average F1, the specified common SHAP cohorts,
> absolute/relative time measures, and the hour/day sweep. Test the setup and
> checkpoint recovery, then run and verify the full experiment. Preserve inputs
> and application settings. Treat historical results as reference evidence,
> not targets to force. Report missing or incompatible prerequisites before
> making substitutions. Do not reset the branch or delete original artifacts.

This brief describes future work. The present authorization is to create this
Markdown guide only; wait for the user's later confirmation before acting on it.
