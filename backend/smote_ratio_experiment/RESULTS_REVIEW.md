# Review of the completed ordinary-SMOTE ratio experiment

Reviewed on 2026-11-16 (Asia/Manila).
Run: `ordinary_smote_ratios_20261116_retry1`.

## Conclusion

No numerical inconsistencies were found in the checks below. Reducing oversampling
improved RF-SMOTE's validation F1 compared with the original 1:100 model. The 1:500
candidate was the strongest of the three tested candidates by F1, but it did not
surpass RF at either the fixed 43% cutoff or their selected shared 40% cutoff.

These are exploratory results on previously inspected validation data. The small
F1 difference at 40% does not establish statistical equivalence or generalization.
No new fits, deployment changes, or test-set evaluation were performed during review.

## Same original 43% cutoff

| Model | Precision | Recall | F1 | Fraud detected | False alarms |
|---|---:|---:|---:|---:|---:|
| RF | 0.837209 | 0.306569 | 0.448798 | 252 | 49 |
| RF-SMOTE 1:100 | 0.427471 | 0.405109 | 0.415990 | 333 | 446 |
| RF-SMOTE 1:200 | 0.543796 | 0.362530 | 0.435036 | 298 | 250 |
| RF-SMOTE 1:300 | 0.628378 | 0.339416 | 0.440758 | 279 | 165 |
| RF-SMOTE 1:500 | 0.748555 | 0.315085 | 0.443493 | 259 | 87 |

At this constant cutoff, moving from 1:100 to 1:500 removes 359 false alarms but
also loses 74 fraud detections overall. These are net count changes, not a claim
that the models' detected record sets are nested.

## Selected shared-cutoff comparisons

| Candidate ratio | Common cutoff for RF and candidate | RF F1 | Candidate F1 |
|---|---:|---:|---:|
| 1:200 | 41% | 0.452697 | 0.435658 |
| 1:300 | 41% | 0.452697 | 0.436923 |
| 1:500 | 40% | 0.450704 | 0.450042 |

The selection procedure correctly maximizes the pair's **mean F1**, not each
model's individual F1. Consequently, 1:300 has lower candidate F1 at the selected
41% cutoff than at 43%; this is expected under the agreed objective, not a bug.

For the closest comparison, both models at 40%:

| Metric | RF | RF-SMOTE 1:500 |
|---|---:|---:|
| Precision | 0.815287 | 0.726287 |
| Recall | 0.311436 | 0.326034 |
| F1 | 0.450704 | 0.450042 |
| MCC | 0.503545 | 0.486205 |
| Average Precision | 0.391528 | 0.388550 |
| Fraud detections | 256 | 268 |
| False alarms | 58 | 101 |

RF-SMOTE has higher recall, but RF retains the higher precision, F1, MCC and
Average Precision. The F1 difference is about 0.00066224 (0.0662 percentage points
when F1 is expressed as a percentage). Do not compare candidate F1 at 40% with RF
F1 at 43% to claim superiority.

## Residual error patterns at the shared 40% cutoff

| Transaction type | RF F1 | 1:500 F1 | Net additional fraud detections | Net additional false alarms |
|---|---:|---:|---:|---:|
| CASH_OUT | 0.554422 | 0.560261 | 9 | 17 |
| TRANSFER | 0.339416 | 0.332756 | 3 | 26 |

TRANSFER has the weaker observed tradeoff. Of the 45 legitimate transactions that
only RF-SMOTE flags, 27 are TRANSFER and 18 are CASH_OUT. RF-SMOTE also avoids two
RF-only false alarms, yielding 43 additional false alarms overall. The candidate
detects all 256 fraud cases detected by RF and 12 additional cases.

Of those 45 RF-SMOTE-only false alarms, 20 are in the 100,000–<1,000,000 amount
band and 18 are at least 1,000,000 dataset units. These counts are descriptive,
not relative-risk estimates or proof that SMOTE caused individual errors.

## Verification performed

- Independent saved-run verifier passed: three fitted candidates; 636,262 matching
  validation identities/labels; full candidate score replay; original reference
  confusion counts; all 100 threshold candidates and the exact shared-cutoff optimum;
  deterministic sampling; 458 explanation score/additivity checks; unchanged inputs,
  output fingerprints and implementation fingerprints.
- Independently recomputed all 18 candidate-comparison overall metric rows and
  **2,268 subgroup metric rows**, including undefined reasons, metric supports and
  sparse flags. Average Precision was recalculated directly from score-tied ranks,
  without using the shared backend metric function.
- Reconciled **30 outcome tables**, all eight outcome classes, amount-band
  boundaries, simulated hour/day derivations, and subgroup totals.
- Compared **3,817,572 exported prediction rows** across the six CSV/Parquet pairs
  (the same 636,262 validation records repeated across three ratios and two cutoffs).
  Values and decisions match. CSV type inference of the mode label `43` was
  normalized as text for comparison; this is not a result discrepancy.
- Checked **1,193 displayed table rows across all four HTML reports** against the
  source CSVs, accounting for intentional six-significant-digit display rounding.
- Verified identical saved preprocessing, fixed forest parameters and seeds, and
  the actual synthetic feature files: 18,847 / 10,375 / 3,597 generated records.
  The fractional-indicator and multiple-positive-type counts are zero for these
  three runs and match their audits.
- Confirmed that all 458 explanations share one background containing exactly 200
  unique original-training records, seed 42, with matching cache fingerprints.
- The existing completed-run browser QA passes offline for desktop/mobile and all
  three detailed candidate reports. The actual desktop report and paired 1:500
  case screenshot were visually inspected during review.

## Interpretation and artifact notes

- Average Precision evaluates score ranking and does not change with the selected
  cutoff. Its improvement over the 1:100 reference is useful, but it remains below RF.
- Model `configuration.json` retains the original 43% scoring setting; the separate
  `comparison.json` records the selected experimental cutoff. The reports explicitly
  include both evaluations. Reading only model configuration will not reproduce the
  selected-cutoff table.
- In combined metric CSVs, `ratio_denominator` identifies the **comparison's
  candidate ratio**. It is not a training-ratio declaration for every row: RF has no
  SMOTE ratio and the row labeled `RF-SMOTE 1:100 reference` remains the 1:100 model.
- The inherited forest `tuning_procedure` string describes the source selected
  configuration; the enclosing isolated manifest records this experiment's actual
  three-ratio protocol. This experiment did not rerun the original three-stage search.
- No additional training or threshold changes are justified automatically by this
  review. Discuss the observed F1 and recall/false-alarm tradeoffs before a new scope.

## Evidence

- [Completed report](outputs/ordinary_smote_ratios_20261116_retry1/report.html)
- [Generated summary](outputs/ordinary_smote_ratios_20261116_retry1/SUMMARY.md)
- [1:500 detailed report](outputs/ordinary_smote_ratios_20261116_retry1/ratio_500/report.html)
- [Detailed independent metric review](runtime/qa/ordinary_smote_ratios_20261116_retry1/detailed_metrics_review.json)
- [Sampling and SHAP background review](runtime/qa/ordinary_smote_ratios_20261116_retry1/sampling_background_review.json)
- [HTML table review](runtime/qa/ordinary_smote_ratios_20261116_retry1/html_table_review.json)

Original completed-run outputs were not edited during this review. Review notes and
additional check results are stored separately from the fingerprinted run.
