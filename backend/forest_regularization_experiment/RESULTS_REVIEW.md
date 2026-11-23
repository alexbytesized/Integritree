# Completed forest regularization experiment

This review describes the fixed-43% experiment. The subsequent
[per-pair cutoff analysis](CUTOFF_RESULTS_REVIEW.md) evaluates each configuration
at its own selected shared cutoff while preserving these original results.

Run: `forest_regularization_20261117`. Reviewed on 2026-11-17 (Asia/Manila).

**All six new fits and all six paired configurations completed. Independent
verification passed. No active model, application setting, or original artifact
was changed, and no held-out test evaluation was performed.**

## Conclusion

The tested restrictions reduced absolute time contributions on the fixed samples
of baseline-flagged records. They did not provide a clear improvement for the
paired RF/RF-SMOTE study:

- Every alternative reduced RF's recall, F1, and Average Precision.
- Every alternative reduced RF-SMOTE's recall and Average Precision.
- RF-SMOTE with depth 10 / leaf 10 was the only alternative with higher F1:
  0.415990 to 0.419375 at the fixed 43% cutoff. It detected four fewer fraud cases
  overall and produced 28 fewer false alarms. RF became worse at that same setting.
- On RF-SMOTE's fixed baseline-flagged sample, every alternative increased time's
  relative share of absolute SHAP contributions, even though its absolute
  magnitude decreased. The original share was 48.75%; alternatives ranged from
  49.02% to 51.69%.
- The baseline retained the highest mean F1 and mean Average Precision across the
  two branches among these six configurations.

These results do not justify replacing the active pair solely to make the time
bars smaller. Retain the current pair while reviewing the tradeoffs. This is an
interpretation of the exploratory experiment, not a claim that the current pair
is optimal or validated for real receipts.

## Protocol and completion

All models used the same 636,262 validation transactions, including 822 labeled
fraud cases. New fits used all 5,090,096 original training rows. Features,
preprocessing, seeds, 100 trees, the final 1:100 ordinary-SMOTE ratio, and the
common 43% cutoff were fixed. Only maximum depth and minimum leaf size varied.

The baseline and the depth-10 leaf-10/leaf-50 model pairs were reused after
verification. Six new forests were fitted: both branches for (8,1), (6,1), and
(8,10). One training-only SMOTE resample added 44,265 synthetic fraud examples
and was shared by all new RF-SMOTE fits. There was no threshold optimization,
seed search, feature removal, or automatic model promotion.

## Full-validation performance at the fixed 43% cutoff

TP is fraud detected; FP is legitimate transactions flagged. Precision, recall,
F1, and AP below are on a 0..1 scale. AP is PR-AUC calculated as Average Precision.

| Model | Depth | Leaf | TP | FP | Precision | Recall | F1 | AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RF baseline | 10 | 1 | 252 | 49 | 0.837209 | 0.306569 | 0.448798 | 0.391528 |
| RF | 8 | 1 | 206 | 32 | 0.865546 | 0.250608 | 0.388679 | 0.375801 |
| RF | 6 | 1 | 141 | 12 | 0.921569 | 0.171533 | 0.289231 | 0.351253 |
| RF | 10 | 10 | 236 | 51 | 0.822300 | 0.287105 | 0.425609 | 0.384432 |
| RF | 10 | 50 | 217 | 50 | 0.812734 | 0.263990 | 0.398531 | 0.370951 |
| RF | 8 | 10 | 195 | 33 | 0.855263 | 0.237226 | 0.371429 | 0.373221 |
| RF-SMOTE baseline | 10 | 1 | 333 | 446 | 0.427471 | 0.405109 | 0.415990 | 0.381196 |
| RF-SMOTE | 8 | 1 | 324 | 460 | 0.413265 | 0.394161 | 0.403487 | 0.367437 |
| RF-SMOTE | 6 | 1 | 286 | 302 | 0.486395 | 0.347932 | 0.405674 | 0.346522 |
| RF-SMOTE | 10 | 10 | 329 | 418 | 0.440428 | 0.400243 | 0.419375 | 0.374883 |
| RF-SMOTE | 10 | 50 | 327 | 450 | 0.420849 | 0.397810 | 0.409006 | 0.373207 |
| RF-SMOTE | 8 | 10 | 325 | 452 | 0.418275 | 0.395377 | 0.406504 | 0.362475 |

Depth 6 illustrates the tradeoff most clearly for RF: false alarms fall from 49
to 12, but fraud detections fall from 252 to 141 and F1 from 0.449 to 0.289.
Lower false alarms and smaller time contributions do not establish improved
fraud detection. These are net count differences; the detected record sets are
not assumed to be nested.

The small F1 gain for RF-SMOTE at leaf 10 is about 0.003385, or 0.339 percentage
points when F1 is expressed as a percentage. It is not evidence of statistical
significance. Its AP falls from 0.381196 to 0.374883, so the ranking metric does
not improve along with this cutoff-specific F1 result.

## Matched SHAP results

Every model used the same 200 original-training background records and the same
387 unique validation explanation records. These comprise overlapping cohorts:
200 uniformly sampled validation records, 100 actual fraud records, and 50
records flagged by each original baseline branch. Cohort membership was frozen
before inspecting any alternative.

The table below uses **each branch's own 50 baseline-flagged records**. Those
records are identical across configurations within a branch, but RF and
RF-SMOTE have different cohorts. Compare changes within each model column;
differences between branches cannot be attributed to model choice alone.

Time magnitude is mean `100 * (abs(hour_SHAP) + abs(day_SHAP))`, measured in
score percentage points. Time share is the cohort sum of absolute time
contributions divided by the sum of absolute contributions across all features.
It is not a percentage of the final score or a causal fraction of fraud.

| Depth / leaf | RF time magnitude | RF time share | RF-SMOTE time magnitude | RF-SMOTE time share |
|---|---:|---:|---:|---:|
| 10 / 1 baseline | 19.73 | 24.66% | 36.96 | 48.75% |
| 8 / 1 | 16.03 | 22.42% | 33.57 | 49.02% |
| 6 / 1 | 11.58 | 20.63% | 30.11 | 51.69% |
| 10 / 10 | 17.25 | 23.58% | 36.30 | 49.37% |
| 10 / 50 | 14.81 | 22.13% | 36.39 | 50.56% |
| 8 / 10 | 15.06 | 22.17% | 33.26 | 49.69% |

The leaf-10 setting barely reduces RF-SMOTE's absolute time magnitude: 36.96 to
36.30 points, about 1.8%. It does not resolve the concern about time dominating
the explanation in this sampled cohort. Depth 6 reduces the absolute magnitude
more strongly, but time's relative share rises to 51.69% as other contributions
also change.

The uniformly sampled baseline cohort looks very different: mean absolute time
magnitude is about 0.12 points for RF and 0.45 for RF-SMOTE, with shares of 6.71%
and 12.54%. This is not a contradiction. The uniform cohort is dominated by
legitimate transactions; flagged records represent a different part of the
distribution. The flagged-case concern should not be dismissed using only a
global average. Full results for all four cohorts are in `time_shap.csv`.

## Time substitution diagnostic

For each of 50 fixed uniformly sampled validation records, all 168 simulation
hour/day combinations were scored with other features held fixed. This produces
8,400 diagnostic scores per model. These artificial combinations can be outside
the observed distribution and carry no new fraud labels; they are not accuracy
tests or evidence about real clock times.

| Depth / leaf | RF mean score range (pp) | RF crossing 43% | RF-SMOTE mean score range (pp) | RF-SMOTE crossing 43% |
|---|---:|---:|---:|---:|
| 10 / 1 baseline | 21.06 | 36% | 31.09 | 36% |
| 8 / 1 | 14.58 | 6% | 25.55 | 36% |
| 6 / 1 | 7.76 | 0% | 24.28 | 36% |
| 10 / 10 | 18.64 | 10% | 30.72 | 36% |
| 10 / 50 | 13.67 | 6% | 29.62 | 36% |
| 8 / 10 | 12.98 | 4% | 25.26 | 36% |

Mean score ranges decreased, but all RF-SMOTE configurations still had 18 of
the 50 sampled records cross the decision boundary under some time substitution.
Lower RF crossing rates must be interpreted alongside its lower fraud recall:
a model that seldom reaches the cutoff can appear less sensitive without being
a better detector. Means and medians behave differently; both are retained in
the full output table.

## Verification and evidence

Independent verification replayed all 12 models on validation data and recomputed
the metric tables, baseline differences, sampling, SHAP summaries, and time
sweeps. It checked:

- 7,635,144 saved validation scores against their fitted models, plus exact labels
  and decisions at the 43% cutoff.
- All 48 cohort-level SHAP summary rows and 4,644 saved explanations, including
  reconstruction of scores from base values and feature contributions.
- All 100,800 time-substitution scores against the models and saved scaler,
  per-record ranges, and decision-boundary crossing rates.
- Reproduction of the 200-row original-training background and all diagnostic
  cohort memberships, with no test-feature reads.
- Source/code/output fingerprints, original baseline report parity, shared SMOTE
  audit and synthetic-row counts, and model parameter consistency.

The tradeoff chart was rendered and visually inspected. The completed run remains
unchanged; this review and additional verification are stored separately.

- [Full offline report](outputs/forest_regularization_20261117/report.html)
- [Metrics](outputs/forest_regularization_20261117/metrics.csv)
- [Changes versus baseline](outputs/forest_regularization_20261117/changes_vs_baseline.csv)
- [All SHAP cohorts](outputs/forest_regularization_20261117/time_shap.csv)
- [Time sensitivity](outputs/forest_regularization_20261117/time_sensitivity.csv)
- [Completion manifest](outputs/forest_regularization_20261117/manifest.json)
- [Independent verification](runtime/qa/forest_regularization_20261117/verification.json)
- [Tradeoff chart](runtime/qa/forest_regularization_20261117/regularization_tradeoff.png)

## Limits on interpretation

These are exploratory results on validation data previously used for model
selection and investigation. There is one seed and small explanation/sensitivity
samples. No significance or generalization claim follows from these comparisons.
The 43% cutoff was held fixed to isolate forest changes; whether separately
retuned cutoffs would alter the tradeoffs was not tested. AP still decreased for
every alternative in both branches.

This experiment changes model complexity. It does not repair the meaning mismatch
between simulated weekday indexes and real receipt weekdays, change the training
features, or validate PaySim patterns for actual GCash fraud detection. None of
the experimental candidates has been deployed.
