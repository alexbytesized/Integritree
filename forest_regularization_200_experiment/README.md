# 200-tree forest regularization experiment

The run `forest_200_20261118` completed on **November 22, 2026**, with all six
new models trained and all 12 models evaluated. Read the
[results table](outputs/forest_200_20261118/SUMMARY.md) or
[offline report](outputs/forest_200_20261118/report.html).

The best shared-cutoff pair by average validation F1 is **200 trees / depth 16 /
leaf 1**, at **48%**, with average F1 **0.644423**. The depth-20 / leaf-1 baseline
selects 59% and achieves average F1 0.633590. No model or cutoff was deployed.
The eight-test suite passed, including training and evaluation resume checks;
the full run verified all 600 cutoff choices, 7,635,144 saved predictions,
4,632 SHAP explanations, and 100,800 time-sweep scores. All 98 final output
fingerprints passed verification, and the report passed desktop/mobile checks.

Training had stopped on November 18 during the last fit. On November 22, the
five completed model checkpoints were verified and reused, and only the final
RF-SMOTE fit was restarted (29.5 minutes). Recorded wall times for the six
successful fits total about 297.9 minutes; this excludes the unfinished attempt.
The manifest's `elapsed_seconds` records the finalized resume session, not the
entire experiment across the interrupted session and intervening days.

Isolated comparison using 200 trees, baseline depth 20 / minimum leaf size 1,
the original prepared training split, and `Experiment-Dataset.csv` for validation.
All new code, models, results, tests, and runtime files stay in this directory.
Existing backend code and saved models are read-only dependencies. Nothing is
promoted to the application, and the official test feature file is not accessed.

## Configurations

| Depth | Minimum leaf | Model source |
|---:|---:|---|
| 20 | 1 | Saved RF / RF-SMOTE baseline |
| 16 | 1 | Two new fits |
| 12 | 1 | Two new fits |
| 20 | 10 | Saved pair |
| 20 | 50 | Saved pair |
| 16 | 10 | Two new fits |

Every model uses 200 trees, seed 42, and one worker. Other forest parameters are
inherited from the saved baseline. All new RF models use the original training
split; all new RF-SMOTE models share one ordinary-SMOTE training sample with
five neighbors and a final 1:100 fraud-to-legitimate ratio. The synthetic tail is
saved once and reused on resume. There is no fitting or resampling on validation.

`config.json` specifies input locations. CSV features must already use the saved
training representation. Only format, numeric finiteness, and binary-label
checks are applied; no provenance/overlap investigation is performed. Rows and
duplicates are preserved; the label never enters model inputs. CSV row numbers
are one-based data-row positions, namespaced by the CSV SHA256 in the manifest.

## Run from the repository root

Preflight verifies sources and compatibility, checks CSV format, and reports
memory availability without training or computing SHAP:

```powershell
& .\backend\.venv\Scripts\python.exe -B .\forest_regularization_200_experiment\run.py
```

Execute training and the full comparison:

```powershell
& .\backend\.venv\Scripts\python.exe -B .\forest_regularization_200_experiment\run.py --execute --run-id forest_200_20261118
```

Omit `--run-id` for a unique new run. Existing run IDs cannot be overwritten.
Use `--config PATH` to change input locations; the experimental grid and sampling
protocol are fixed. Execution will be substantially longer than evaluation-only
runs because six models need training.

Resume an unfinished run:

```powershell
& .\backend\.venv\Scripts\python.exe -B .\forest_regularization_200_experiment\run.py --execute --resume forest_200_20261118
```

Resume requires unchanged input/configuration/code fingerprints and dependency
versions. Completed model files are verified before reuse. Interrupted fits
restart; completed fits do not. Completed pair evaluations also resume from
verified checkpoints. An unfinished pair evaluation is recalculated. Temporary
files never count as completion markers. A `.run.lock` blocks concurrent writers;
after a hard process termination, verify that the old process has stopped before
removing only that run's lock. A completed run is immutable to this runner.

## Results and interpretation

Each output directory contains:

- `models/`: the six newly fitted forests with fingerprinted reload-verification
  checkpoints. Reused models remain at their original paths.
- `evaluation/`: full CSV scores, fixed and selected decisions, 100-cutoff tables,
  SHAP vectors, hour/day sweeps, per-record sensitivity, and pair checkpoints.
- `comparison.csv`, `SUMMARY.md`, `report.html`: precision, recall, F1, pair average
  F1, PR-AUC (average precision), fraud detected, false alarms, and time SHAP.
- `changes_vs_200_tree_baseline.csv`: differences from the new baseline under each
  cutoff policy. Selected-policy comparisons may use different pair cutoffs.
- `previous_100_tree_reference.csv`: separately labeled earlier results. Tree
  count and depth both differ, so this is not an isolated depth comparison.
- `time_shap.csv`, `time_sensitivity.csv`, cohort files, and the 200-record original
  training background: reproducible diagnostic evidence.
- `manifest.json`, `plan.json`, `training_progress.json`, `run.log`, and
  `verification.json`: status, timings, exact inputs, code/package fingerprints,
  resource checks, checkpoint progress, and verification evidence.

Every pair is evaluated at 43% and its own best shared cutoff on the 1%-100% grid.
The objective is average RF/RF-SMOTE F1; exact ties prefer closest to 50%, then
higher. A score equal to the cutoff predicts fraud. Shared optima need not be
optimal for individual models. These are experimental validation results:
the CSV used to choose cutoffs is also used to report their performance.

SHAP uses 200 uniformly sampled CSV records, 100 fraud records, and up to 50
records flagged by each new baseline model at 43%. Cohorts overlap and are frozen
across models. The background is 200 original training records, sampled with
seed 42. The SHAP kernel receives a float32 background to match sklearn's input
representation at split boundaries; prediction-equivalence and score
reconstruction checks remain enforced. The actual saved model is not altered.

Displayed time contribution is `100 * mean(abs(SHAP hour) + abs(SHAP day))` in
score percentage points. Time share is the ratio of summed absolute time SHAP
to summed absolute SHAP for all features. Main tables use the uniform cohort.
The 50-record sensitivity sample substitutes all 168 hour/day combinations;
these substitutions are diagnostics, not labeled transactions or accuracy tests.

## Verification

```powershell
& .\backend\.venv\Scripts\python.exe -B -m pytest -c .\forest_regularization_200_experiment\pytest.ini .\forest_regularization_200_experiment\tests -q
& .\backend\.venv\Scripts\python.exe -B .\forest_regularization_200_experiment\verify.py .\forest_regularization_200_experiment\outputs\forest_200_20261118
```

Tests exercise real synthetic training and SHAP, interrupted-fit recovery,
evaluation checkpoint recovery, corruption/context rejection, training-only
SMOTE reuse, CSV preservation, fixed protocol settings, and preflight without
training. The verifier recomputes reported metrics and all 600 threshold choices
from saved scores and checks row alignment, SHAP summaries/reconstruction, pair
averages, time sweeps, and source/output fingerprints.
