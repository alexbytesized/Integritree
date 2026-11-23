# Experimental validation results — November 18, 2026

All three stages completed using the supplied `Experiment-Dataset.csv`: **636,262 records**, including **822 fraud labels**. The run took approximately **223 seconds**, reused saved models, and performed **zero training fits**.

| Stage | Selection | Evidence |
|---|---|---|
| 1: SMOTE ratio | **1:100** | Highest RF-SMOTE AP: **0.606128294**; runner-up 1:50: **0.602956851**. |
| 2: forest | **200 trees, maximum depth 20, minimum leaf size 10** | Highest mean RF/RF-SMOTE AP: **0.613839530**. |
| 3: common cutoff | **44%** | Highest mean F1 on the 1%–100% grid: **0.608642752**. |

Stage 1 therefore retained the original ratio, allowing all 12 saved forest configurations at 1:100 to be compared. The next-best forest was 100 trees/depth 20/minimum leaf 10, with mean AP **0.613754910**. The selected configuration's lead is small; the comparison uses one experimental validation dataset.

At the selected shared 44% cutoff:

| Model | Precision | Recall | F1 | Average Precision | TP | FP | TN | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RF | 0.882086 | 0.473236 | 0.615994 | 0.615826 | 389 | 52 | 635,388 | 433 |
| RF-SMOTE | 0.581158 | 0.622871 | 0.601292 | 0.611853 | 512 | 369 | 635,071 | 310 |

The forest selection objective was mean **AP**, not mean F1. The cutoff was then optimized for the selected forest pair. These results do not establish that this pair has the highest achievable F1 across every forest/cutoff combination.

## Verification

- **14 tests passed**, including complete and missing-model paths with training and resampling forbidden.
- Independent verification passed for **19 distinct candidate pairs**, their metrics and selections, label/row alignment, final decisions, and **all 100 cutoff confusion matrices**. Exact selected mean F1: `1309123/2150889`.
- Input, code and model fingerprints remained unchanged during execution.
- Offline browser verification passed, including the loaded plot, no remote requests, and no page-level overflow at mobile width.

These are experimental validation results, using the same CSV for selection and reporting. The CSV's independence and overlap have not been established. The active application retains its existing models and cutoff.

## Saved evidence

- [Offline report](outputs/experimental_validation_20261118/report.html)
- [Stage 1: all eight ratios](outputs/experimental_validation_20261118/01_smote_ratio/candidates.csv)
- [Stage 2: all twelve forests](outputs/experimental_validation_20261118/02_random_forest/candidates.csv)
- [Stage 3: all hundred cutoffs](outputs/experimental_validation_20261118/03_threshold/thresholds.csv)
- [Final metrics](outputs/experimental_validation_20261118/final_metrics.csv)
- [Manifest](outputs/experimental_validation_20261118/manifest.json)
- [Run verification](outputs/experimental_validation_20261118/verification.json)
- [Browser verification](runtime/qa/experimental_validation_20261118/browser_verification.json)

Full row-level candidate scores and final predictions are saved as Parquet files in the run directory. See the [README](README.md) for rerun and independent verification commands.
