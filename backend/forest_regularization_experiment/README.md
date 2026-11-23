# Isolated depth and leaf-size experiment

## Repeat evaluation with an engineered CSV

`evaluate_csv.py` reuses all 12 saved models from the completed six-pair run.
It performs no fitting, SMOTE, imputation, scaling, or deduplication. The input
must contain the 11 feature columns plus binary `isFraud`; features must already
use the saved training scaler's representation. Labels are excluded from model
inputs. The supplied CSV's row order and duplicate rows are preserved.

```powershell
& .\backend\.venv\Scripts\python.exe -B .\backend\forest_regularization_experiment\evaluate_csv.py --csv 'C:\Users\Carpicorn\Downloads\Experiment-Dataset.csv' --run-id csv_validation_20261118
```

Omit `--run-id` for a fresh unique output directory on subsequent runs. Existing
directories are never overwritten. `--source` and `--prior-cutoffs` can identify
the completed original forest experiment and its corresponding cutoff results;
their defaults refer to the November 17 runs below.

Results for this CSV are in
[`outputs/csv_validation_20261118`](outputs/csv_validation_20261118):

The November 18 run is complete and verified: 636,262 rows, 822 fraud labels,
12 reused models, zero new fits, and 242.22 seconds elapsed. The current
depth-10 / leaf-1 pair retains the highest selected-cutoff mean F1 (0.618933)
with a newly selected shared cutoff of 46%. This does not change the active
application's 43% cutoff. The experiment test suite passed all 28 tests.

- `SUMMARY.md` and `report.html`: all six pairs at 43% and their newly selected
  shared cutoffs, including precision, recall, F1, average precision, detections,
  false alarms, and uniform-cohort time SHAP measures.
- `comparison.csv`, `changes_vs_previous.csv`, `shap_changes_vs_previous.csv`:
  full-precision metrics and comparisons with the earlier validation results.
- `time_shap.csv`, `time_sensitivity.csv`, `shap_cohorts.csv`: all four overlapping
  SHAP cohorts and time-sweep results. Samples are regenerated from CSV positions
  using seed 42; the original 200-row training background is reused unchanged.
- Each `depth_*` directory: full CSV predictions, fixed and selected decisions,
  all 100 cutoff choices, SHAP explanations, and 168-point time sweeps.
- `manifest.json`, `verification.json`, `run.log`: completion status, input/model
  references and hashes, timings, independent cutoff-count verification, and
  SHAP score-reconstruction checks.

`source_row_number` in CSV-run outputs means the **1-based data-row position in
this CSV**, excluding its header. It is not an original PaySim transaction ID;
the CSV SHA256 in the manifest supplies its dataset namespace. Changing row
order changes sampled records even with the same random seed. SHAP comparisons
with the earlier run therefore compare sample summaries, not matched records.

Cutoffs maximize the pair's mean validation F1 over 1% through 100%, with exact
ties resolved closest to 50%, then higher. Reported metrics use the same data
used to select cutoffs and are experimental validation results. The supplied
dataset's provenance and overlap are not investigated by this runner. No
official held-out test file is accessed and no application settings change.

## Original experiment

The full run `forest_regularization_20261117` is complete and independently
verified. Read the [results review](RESULTS_REVIEW.md) for measured tradeoffs and
links to the report, chart, and verification evidence. No model was promoted.

The subsequent [shared-cutoff comparison](CUTOFF_RESULTS_REVIEW.md) is also
complete: each of the six configurations was evaluated at its own best shared
cutoff on the 1%..100% grid, using saved scores without retraining. The current
pair still selects 43% and retains the highest mean validation F1.

This experiment tests whether simpler Random Forest trees reduce sensitivity to
simulation time while retaining useful fraud detection. It compares RF and
RF-SMOTE under the same conditions. It does not replace the active models, change
the original study configuration, or evaluate the held-out test set.

## Understanding the two settings

`max_depth` limits the number of successive splits along a tree path. A depth-10
tree can form more specific combinations of transaction type, amount, hour, and
day than a depth-6 tree. Lower depth can remove overly specific patterns, but it
can also remove useful interactions and cause underfitting.

`min_samples_leaf` requires enough training samples in both child leaves before
a split is permitted. Increasing it from 1 to 10 or 50 prevents rules supported
by tiny groups. It is **not** the number of fraudulent examples required: both
classes count, and synthetic examples count in the SMOTE branch. It does not
guarantee 50 independent real transactions as evidence for a rule.

Both are training settings. Existing fitted trees do not change when a
configuration file is edited: a candidate must be refitted, or an already fitted
matching candidate can be reused. Neither setting specifically penalizes time.
If time is a strong early split, shallower trees may still rely heavily on it.
Relative SHAP time share can even increase when other effects are removed.

For example, a highly specific rule involving transaction type, amount, hour,
day, and further conditions might disappear in a shallower tree. A simple
first-level hour rule can survive. This example illustrates capacity; it is not
a claim about any particular saved tree.

Reference: [RandomForestClassifier parameter definitions](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).

## Prespecified comparisons

| Candidate | Maximum depth | Minimum leaf size | Purpose |
|---|---:|---:|---|
| Baseline | 10 | 1 | Existing selected model pair |
| Shallower | 8 | 1 | Change only depth |
| Much shallower | 6 | 1 | Stronger depth restriction |
| Larger leaves | 10 | 10 | Change only minimum leaf size |
| Much larger leaves | 10 | 50 | Stronger leaf restriction |
| Combined | 8 | 10 | Test the two restrictions together |

This small design tests individual changes plus one combination; it is not a
complete factorial search. It does not establish that untested combinations are
inferior. The same settings apply to both model branches in every comparison.

Keep fixed: 100 trees, model/SMOTE seed 42, original training and validation
partitions, all 11 predictors, fitted preprocessing, `max_features`, bootstrap,
other forest parameters, ordinary SMOTE with five neighbors and a final 1:100
fraud:legitimate ratio, and the common **43% cutoff**. The ratio experiment's
1:500 candidate is not used here: changing the ratio too would confound the
forest comparison. There is no threshold search or automatic winner selection.

The original study already tested depth 10 with leaf sizes 10 and 50. The runner
reuses those verified artifacts when present. It checks their full forest
parameters, preprocessing, prepared-data provenance, seeds, and SMOTE settings.
The existing application loader checks their hashes and dependency versions.
Reuse only avoids repeating an identical deterministic fit. It does not reuse
old explanations: every pair receives fresh matched diagnostics.

The default local plan has three reused pairs (including baseline) and **six new
fits**: RF and RF-SMOTE for (8,1), (6,1), and (8,10). If nonbaseline saved pairs
are absent, the plan transparently schedules those fits too. Invalid or ambiguous
saved pairs fail verification instead of being silently substituted.

## Run from the repository root in PowerShell

Use the existing backend environment; no installation is required.

Verify inputs and produce a plan, without new training or SHAP calculation:

```powershell
& .\backend\.venv\Scripts\python.exe -B .\backend\forest_regularization_experiment\run.py
```

Run the full comparison:

```powershell
& .\backend\.venv\Scripts\python.exe -B .\backend\forest_regularization_experiment\run.py --execute
```

Both commands print their output folder. Optional `--run-id <unique-name>` gives
a named run. Existing names are rejected, including failed runs. Relative input
paths in `--config <path>` resolve against that configuration file. All outputs
are under this experiment's `outputs/`, regardless of the working directory.

Full execution uses all 5,090,096 training rows and all 636,262 validation rows,
with one training worker and sequential candidates. Training and probability-space
SHAP may take substantial time. No small-data pilot is presented as a full study
result. Before execution, the runner checks the existing conservative RAM
estimate. This is not a guarantee of peak memory. Progress is written to stderr
and `run.log`; `manifest.json` records completed candidates and fit starts.

The first version does not resume interrupted runs. Completed candidate evidence
is preserved in a failed run; use a new run ID for a rerun. A report written
before all candidates finish is a partial checkpoint. Verify `manifest.json`
has `status: complete` before interpreting a full comparison.

## What is measured

1. **Full-validation prediction quality at 43%:** precision, recall, F1, MCC,
   Average Precision (PR-AUC), accuracy, TP/FP/TN/FN, false alarms per 10,000
   legitimate records, and differences from each branch's original baseline.
   Undefined metrics carry explicit reasons. AP measures score ranking and is
   independent of this cutoff. A lower fraud flag count is not itself improvement.
2. **SHAP on identical records and background:** one uniform sample of up to
   200 validation records; up to 100 actual fraud cases; and up to 50 records
   flagged by each original baseline branch. Membership is frozen before any
   candidate is inspected. The overlapping cohorts are reported separately;
   enriched fraud/flagged cohorts do not estimate population importance. Every
   model uses the same 200 unique original-training background rows, seed 42,
   probability-space interventional TreeSHAP, and the existing prediction-equivalent
   SHAP adapter. All features remain in the explanation.
3. **Time sensitivity on a separate fixed uniform sample of 50 validation rows:**
   hold other predictors fixed and score all 168 simulation hour/day combinations.
   Save each score, each record's range in percentage points, and whether any
   substitution crosses 43%. These artificial substitutions can be outside the
   observed joint distribution. They measure model response, not fraud accuracy,
   real-world behavior, or causality.

Every SHAP result must reconstruct the model score within 1e-6. Prediction parity
between the SHAP adapter and sklearn is checked on the background. New models
are saved/reloaded and prediction parity checked. The source files and relevant
implementation files are fingerprinted and checked again at completion.

### Reading the time columns

- `mean_time_abs_pp`: mean of `100 * (abs(hour_SHAP) + abs(day_SHAP))`.
  Measures contribution magnitude without positive/negative cancellation.
- `mean_time_net_abs_pp`: mean of `100 * abs(hour_SHAP + day_SHAP)`.
  Measures the magnitude of their signed sum; opposing contributions can cancel.
- `time_share_abs`: sum of absolute time contributions divided by the sum of
  absolute contributions across all features in that cohort. This is **not**
  the percentage of fraud causally explained by time, nor a fraction of the
  final probability. It is undefined if all contributions are zero.
- `time_strictly_largest_group_fraction`: fraction where time's summed absolute
  contributions exceed each other group (transaction type, amount, entity).
  Ties are not counted. Group sizes and correlated features affect comparisons.

Use absolute magnitudes and score sensitivity together with relative share.
Do not reduce, hide, or rescale SHAP contributions to make a chart look better.
Background/base scores may differ by fitted model despite identical reference
records; the saved explanations contain each base score.

## How to decide after execution

Compare each RF candidate against baseline RF, and each RF-SMOTE candidate against
baseline RF-SMOTE. First read TP/FP changes and recall/precision/F1/AP; then examine
time magnitude, relative share and sensitivity on matched cohorts. Report any
tradeoff. A candidate that removes many false alarms by missing more fraud is not
unconditionally better. A nearly constant low-score model could have tiny time
effects and still be useless for detection.

Do not select by the smallest SHAP bar alone. Review candidates with useful
predictive performance and lower sensitivity, then specify an acceptable loss
of recall or other operational criterion before adopting one. This experiment
does not invent an acceptable loss for the researchers or automatically promote
a model. Retaining the baseline is a valid outcome.

Validation has already been inspected in the original selection and subsequent
investigations. Results are exploratory, not a new unbiased test or statistical
proof of superiority. One seed and small SHAP cohorts do not establish robustness.
Any final selection needs a separately planned, frozen evaluation; no test-set
results are accessed here. Also, tree regularization does not correct the
simulation-weekday/real-receipt-weekday meaning mismatch, and synthetic PaySim
results do not establish real GCash fraud-detection performance.

## Output evidence and isolation

- `plan.json`, `manifest.json`, `run.log`: configuration, source/reuse paths,
  input/code/dependency fingerprints, resource estimate, fit count, and status.
- `report.html`, `SUMMARY.md`: offline summary; no server or network required.
- `metrics.csv`, `changes_vs_baseline.csv`, `time_shap.csv`,
  `time_sensitivity.csv`: machine-readable comparison tables.
- `shap_background.parquet`, `shap_cohorts.csv`, `sensitivity_cohort.csv`:
  identical diagnostic reference records and sample identities.
- `smote_audit.json`, `synthetic_features.parquet`: one shared training-only
  resample for all newly fitted RF-SMOTE candidates.
- `depth_*/`: full validation scores/labels with source row identity, per-model
  SHAP vectors/base scores, all 168-point time sweeps, tree-size diagnostics, and
  new `.joblib` artifacts where fitted. Reused artifacts stay at their recorded
  read-only source paths. Dataset identity is in `plan.json`.

New artifacts have an isolated format that the application bundle loader rejects.
Nothing changes the selected model, original artifacts, active environment,
frontend, receipt processing, or previous investigation/ratio experiment. This
runner imports only the established backend package, not either untracked
experiment. The split manifest contains all partition identities, but only train
and validation records are selected; the test feature file is never opened or hashed.

Run the synthetic verification suite:

```powershell
& .\backend\.venv\Scripts\python.exe -B -m pytest -c .\backend\forest_regularization_experiment\pytest.ini .\backend\forest_regularization_experiment\tests --basetemp .\backend\forest_regularization_experiment\runtime\pytest
```

Tests cover a complete synthetic fit/save/reload/evaluate/SHAP/report cycle,
matched identities, no test-split access, no training in preflight, source
preservation, refusal to overwrite a run, rejected incompatible reuse, and
absolute-versus-net contribution arithmetic. Synthetic test results are software
checks, not measured PaySim experiment findings.
