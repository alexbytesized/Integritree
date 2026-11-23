# Best shared cutoff for each forest pair

Completed on 2026-11-17 (Asia/Manila). Analysis:
`paired_cutoffs_20261117`, using saved scores from
`forest_regularization_20261117`.

**All six pairs, including the current pair, were evaluated at their own best
shared cutoff on the study's 1%..100% grid. No models were retrained. The current
depth-10 / leaf-1 pair still has the highest mean validation F1.**

## Selection rule

For each pair, apply the same cutoff to RF and RF-SMOTE and maximize:

`mean F1 = (RF validation F1 + RF-SMOTE validation F1) / 2`

Candidates are exactly 1%, 2%, ..., 100%. Exact F1 ties prefer the cutoff closest
to 50%, then the higher cutoff. Scores equal to the cutoff predict fraud. Exact
rational comparisons are used for selection, rather than rounded displayed F1.

This is a shared pair optimum among the 100 tested grid values. It is not an
individual optimum for each branch or an optimization over every real-valued
cutoff. All comparisons use the same 636,262 validation records, including 822
fraud labels. No held-out test evaluation or deployment occurred.

## Pair-level results

| Depth / leaf | Best shared cutoff | RF F1 selected | RF-SMOTE F1 selected | Mean F1 at 43% | Mean F1 selected |
|---|---:|---:|---:|---:|---:|
| **10 / 1 (current)** | **43%** | **0.448798** | **0.415990** | **0.432394** | **0.432394** |
| 8 / 1 | 40% | 0.417802 | 0.399044 | 0.396083 | 0.408423 |
| 6 / 1 | 41% | 0.302846 | 0.398658 | 0.347452 | 0.350752 |
| 10 / 10 | 44% | 0.421719 | 0.425393 | 0.422492 | 0.423556 |
| 10 / 50 | 47% | 0.386492 | 0.428571 | 0.403768 | 0.407531 |
| 8 / 10 | 39% | 0.402904 | 0.397398 | 0.388966 | 0.400151 |

The current pair's selected cutoff remains 43%, reproducing the original study
selection. Each alternative improves its mean F1 relative to its own 43% result,
but none exceeds the current pair's 0.432394. The closest alternative is depth
10 / leaf 10 at 44%, with mean F1 0.423556.

Optimizing the pair does not guarantee improvement for both branches. For
example, depth 8 / leaf 1 moves from 43% to 40%: RF F1 increases from 0.388679
to 0.417802, while RF-SMOTE F1 decreases from 0.403487 to 0.399044. The net mean
improves, which is why the declared selection rule chooses 40%.

## Detection results at selected shared cutoffs

| Depth / leaf | Cutoff | Model | Precision | Recall | F1 | Fraud detected | False alarms |
|---|---:|---|---:|---:|---:|---:|---:|
| **10 / 1 current** | **43%** | **RF** | **83.72%** | **30.66%** | **0.448798** | **252** | **49** |
| **10 / 1 current** | **43%** | **RF-SMOTE** | **42.75%** | **40.51%** | **0.415990** | **333** | **446** |
| 8 / 1 | 40% | RF | 82.44% | 27.98% | 0.417802 | 230 | 49 |
| 8 / 1 | 40% | RF-SMOTE | 39.20% | 40.63% | 0.399044 | 334 | 518 |
| 6 / 1 | 41% | RF | 91.98% | 18.13% | 0.302846 | 149 | 13 |
| 6 / 1 | 41% | RF-SMOTE | 44.46% | 36.13% | 0.398658 | 297 | 371 |
| 10 / 10 | 44% | RF | 82.33% | 28.35% | 0.421719 | 233 | 50 |
| 10 / 10 | 44% | RF-SMOTE | 46.03% | 39.54% | 0.425393 | 325 | 381 |
| 10 / 50 | 47% | RF | 84.43% | 25.06% | 0.386492 | 206 | 38 |
| 10 / 50 | 47% | RF-SMOTE | 47.49% | 39.05% | 0.428571 | 321 | 355 |
| 8 / 10 | 39% | RF | 79.29% | 27.01% | 0.402904 | 222 | 58 |
| 8 / 10 | 39% | RF-SMOTE | 38.67% | 40.88% | 0.397398 | 336 | 533 |

RF-SMOTE's strongest selected F1 among these configurations is depth 10 / leaf
50 at 47% (0.428571), compared with current RF-SMOTE at 43% (0.415990). This
comparison includes both a forest change and a cutoff change. That candidate
has 12 fewer fraud detections and 91 fewer false alarms overall; its paired RF
has substantially lower F1 than current RF. It is therefore not the best pair
under the chosen mean-F1 objective.

## Relationship to time and SHAP

Cutoffs change which scores become fraud flags. They do not change fitted trees,
scores, or probability-space SHAP contributions for the same transaction and
background. Average Precision is also unchanged; it evaluates score ranking
across thresholds. The previous absolute-time and relative-time SHAP results
still describe the same fixed original cohorts.

The newly flagged sets can differ. No new flagged-set SHAP sample was generated
in this follow-up, so the old flagged-cohort table must not be relabeled as a
summary of the new flags. `time_boundary_changes.csv` separately records how
the selected cutoffs change boundary crossing on the original 50-record time
substitution sample; the underlying 168-point score sweeps are unchanged.

## Verification, files, and reproduction

Six synthetic checks passed, covering score equality at the cutoff, exact tie
handling, independent-grid rejection of incorrect selections, source corruption,
refusal to overwrite a run, and a complete analysis that forbids fitting.

On the real data, the runner independently counted TP/FP/TN/FN and computed
exact mean F1 at all **600 pair/cutoff combinations**, checking the selected
optimum and tie policy. A separate export verifier checked **7,635,144**
model-record decisions, source score preservation, selected metrics, common
cutoffs, unchanged Average Precision, and all source/implementation/output
fingerprints. Both verification stages passed. No training, model loading,
feature extraction, or test-feature access was needed for cutoff selection.

- [Offline report](outputs/paired_cutoffs_20261117/report.html)
- [Pair summary](outputs/paired_cutoffs_20261117/pair_summary.csv)
- [Selected-cutoff metrics](outputs/paired_cutoffs_20261117/metrics_selected.csv)
- [Fixed-43% metrics](outputs/paired_cutoffs_20261117/metrics_fixed_43.csv)
- [Changes versus each configuration at 43%](outputs/paired_cutoffs_20261117/changes_vs_43.csv)
- [All 600 cutoff candidates](outputs/paired_cutoffs_20261117/threshold_grid.csv)
- [Time-boundary changes](outputs/paired_cutoffs_20261117/time_boundary_changes.csv)
- [Direct grid verification](outputs/paired_cutoffs_20261117/verification.json)
- [Independent export verification](runtime/qa/paired_cutoffs_20261117/verification.json)
- [Manifest](outputs/paired_cutoffs_20261117/manifest.json)

Each candidate subfolder contains its decision, full threshold curve, and saved
validation scores with both the original 43% labels and newly selected labels.
CSV and Parquet tables are provided; use Parquet or CSV parsing with round-trip
float precision when replaying exact score/cutoff decisions.

To repeat into a new, uniquely named output folder from the repository root:

```powershell
& .\backend\.venv\Scripts\python.exe -B .\backend\forest_regularization_experiment\compare_cutoffs.py
```

`--source` accepts another completed isolated forest experiment; `--run-id`
accepts an unused output name. Existing output directories are never overwritten.

## Interpretation limits

These are optimized results on previously inspected validation data, not new
independent test performance or proof of significance. The threshold search
does not resolve the simulation-time versus real-receipt-time mismatch. The
active pair and its 43% cutoff remain unchanged. The results continue to favor
the current pair by the declared mean-F1 criterion among the tested settings.
