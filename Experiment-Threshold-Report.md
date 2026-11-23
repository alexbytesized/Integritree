# Experimental dataset threshold results

Analysis date: 18 November 2026.

**The best shared cutoff on this dataset is 0.46841 (46.841%), maximizing the arithmetic mean of RF and RF-SMOTE positive-class F1 to 0.620798.** Classify a transaction as fraud when its unrounded model score is greater than or equal to the cutoff.

**Use 0.46 (46%) if following Integritree's existing 1%, 2%, ..., 100% threshold protocol.** Its mean F1 is 0.618933. The finer result is an additional exploratory analysis outside that grid. These are dataset-specific selection results, not independent test estimates.

## Cutoff comparison

| Selection | Shared cutoff | RF F1 | RF-SMOTE F1 | Mean F1 |
|---|---:|---:|---:|---:|
| Existing cutoff | 0.43 | 0.607656 | 0.622431 | 0.615043 |
| Best whole percentage | 0.46 | 0.591389 | 0.646477 | 0.618933 |
| Best unrestricted cutoff | 0.46841 | 0.590724 | 0.650873 | 0.620798 |
| Default comparison | 0.5 | 0.559933 | 0.655194 | 0.607564 |

The finer optimum improves mean F1 over 46% by 0.1865 percentage points. This numerical improvement is not evidence of a statistically significant or future-performance improvement. Increasing the cutoff trades recall for fewer false positives; no error-cost or minimum-recall requirement was supplied.

## Performance at 46.841%

| Model | Precision | Recall | F1 | True positives | False positives | True negatives | False negatives |
|---|---:|---:|---:|---:|---:|---:|---:|
| RF | 89.1892% | 44.1606% | 0.590724 | 363 | 44 | 635,396 | 459 |
| RF-SMOTE | 66.7519% | 63.5036% | 0.650873 | 522 | 260 | 635,180 | 300 |

The objective weights the two models equally. It is not macro-F1 over fraud and legitimate classes, and it does not independently maximize each model's F1.

## Method and reproducibility

- Source: `C:/Users/Carpicorn/Downloads/Experiment-Dataset.csv`. SHA-256: `4a883825dfe58a8f9e445131e06c4251ce2c7fa9e8a29e1490f667b27d961b1d`.
- Analyzed all 636,262 rows: 822 fraud and 635,440 legitimate transactions. Fraud prevalence: 0.129192%.
- All 11 model features were present in the expected order. There were no missing cells or non-finite feature values, and labels were binary. Features were already scaled and were passed directly to the saved models without scaling a second time. There were 407 repeated feature-and-label rows; all were retained because these features do not uniquely identify transactions.
- Used the existing selected pair from `paysim_three_stage_20260928_172539`, candidate `ratio_0.01_trees_100_depth_10_leaf_1`, training run `train_20260928T092543_93f9a4e8`. Both forests use 100 trees, maximum depth 10, minimum leaf size 1, and seed 42. RF-SMOTE uses ordinary SMOTE with final fraud-to-legitimate ratio 1:100. Model files and their metadata were checked against stored SHA-256 fingerprints before loading.
- Computed class-1 probabilities using the saved forests. No model fitting, resampling, probability calibration, or parameter changes were performed.
- Evaluated all 100 whole-percentage cutoffs using the existing project threshold function. Exact grid ties favor the cutoff closest to 0.50, then the higher cutoff. Independently verified every grid confusion matrix and F1 against direct predictions.
- Also evaluated 30,000 candidates covering the union of both models' distinct scores and endpoints 0 and 1. This covers every distinct shared classification outcome for thresholds in [0, 1]. Potential maxima were compared using exact rational F1. The optimal interval is **(0.46823129190697799, 0.46858837055814689]**. Every cutoff in that interval produces identical predictions; 0.46841 is a convenient interior value. Directly checked its metrics.
- Mean F1 at the fine optimum is exactly `611895/985658`. Complete settings, file fingerprints, package versions, and counts are in `Experiment-Threshold-Results.json`.

## Data overlap and interpretation

A multiset comparison with the existing prepared validation data matched **635,952 of 636,262 rows (99.9513%)**, using all features and the label rounded to 12 decimal places and pandas row hashes. The supplied file is not an exact match; 310 row occurrences were unmatched at that precision. This comparison establishes extensive feature/label overlap, not original transaction identity, because the supplied file has no source-row identifier. The cause of the unmatched rows was not determined.

The models and original 43% cutoff were selected using the existing validation data. This new threshold was selected and measured on the supplied data itself. Consequently, the reported metrics cannot be presented as results on an independent unseen test set. Keep the original validation result separate from this experiment, and evaluate any adopted cutoff on independent labeled data before claiming generalization. No separate held-out test data was used in this analysis.

The existing model artifacts, project configuration, and official validation reports were not changed. The experimental threshold was not applied to the application. No one-off analysis script was retained.

## Optional separate cutoffs

If the research design permits a different cutoff for each model, the individual F1 maxima on these same supplied rows are:

| Model | Individual cutoff | F1 |
|---|---:|---:|
| RF | 0.15938945591422946 | 0.657462 |
| RF-SMOTE | 0.51154209461630418 | 0.658552 |

These separate cutoffs answer a different question from Integritree's shared-threshold comparison. They are included for context and were not used as the shared recommendation.

## Saved results

- `Experiment-Threshold-Sweep.csv`: all 100 whole-percentage cutoffs, model metrics, confusion counts, and mean F1. Blank precision or MCC means undefined at that cutoff.
- `Experiment-Threshold-Results.json`: exact optimum, grid optimum, reference cutoffs, provenance, and supplementary individual optima.
- `Experiment-Threshold-Plot.png`: whole-percentage F1 curves and highlighted cutoffs.
