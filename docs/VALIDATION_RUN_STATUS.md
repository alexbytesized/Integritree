# Validation run status

## Current status: Stage 3 complete

Run: `paysim_three_stage_20260928_172539`.

The authorized Stage 3 rerun completed on **2026-09-29**,
from **04:05:56 to 04:07:02 Manila time**
(66 seconds; process exit code 0). No models were retrained.

**Selected shared fraud threshold: 43% (0.43).**
Both models classify a transaction as fraud when its score is at least this cutoff.

Stage 3 evaluated exactly **100 thresholds: 1%, 2%, ..., 100%**, maximizing
mean validation F1 across RF and RF-SMOTE. Exact ties favor proximity to 50%,
then the higher percentage. The result is optimal among these tested percentages.

| Validation metric | RF | RF-SMOTE |
|---|---:|---:|
| Precision | 0.837209 | 0.427471 |
| Recall | 0.306569 | 0.405109 |
| F1 | 0.448798 | 0.415990 |
| MCC | 0.506284 | 0.415405 |
| Average Precision | 0.391528 | 0.381196 |

Mean F1: **0.432393935**, exactly `777411/1797923`.
Evaluation used **636,262 validation transactions**, including 822 fraud records.
The manifest records `status: complete`, `stage: validation_selected`,
`completed_stage: 3`, and `test_used: false`. Official test evaluation and SHAP
have not run on this selected pair.

## Preserved Stages 1 and 2

Stage 1 completed all eight ratios on 2026-09-28 and selected **1:100**.
Stage 2 completed all 12 shared forest configurations at 03:06 Manila time on
2026-09-29 and selected **100 trees, depth 10, minimum leaf size 1**.

All **351 preserved files** passed unchanged SHA-256 checks, covering fitted
models, candidate validation evidence, candidate tables, decisions, and the
progress checkpoint. The final validation scores exactly match the saved scores
for the selected pair.

## Protocol revision and verification

The threshold protocol was revised after validation inspection. Its implementation,
configuration, tests, and documentation now support only the 1%-100% grid. Prior
Stage 3 results and logs were removed; no result archive was retained. Stage 1-2
review snapshots and checksums were regenerated under the revised protocol and
are explicitly labeled as regenerated. A reset audit records cleanup actions and
preservation hashes. Existing Git and conversation history were not rewritten.

The superseded preflight-failed run remains a failure record; its unused Stage 3
declarations were removed, and it must not be resumed. Prepared data and original
Word reference documents were not modified.

Verification passed: selected-bundle loading, exhaustive direct comparison of all
100 thresholds, exported metric/statistical recomputation, report fingerprints,
and unchanged candidate scores. All 100 rows are saved and plotted. The backend
suite passed **186 tests** (five upstream deprecation warnings); **17 focused
checks** passed after the final table/plot formatting change. `pip check` passed.

## Results and evidence

- [Validation summary](../backend/reports/paysim_three_stage_20260928_172539/SUMMARY.md)
- [Final metrics](../backend/reports/paysim_three_stage_20260928_172539/final_validation/metrics.json)
- [Threshold table](../backend/reports/paysim_three_stage_20260928_172539/stages/03_threshold/thresholds.csv)
- [Threshold plot](../backend/reports/paysim_three_stage_20260928_172539/stages/03_threshold/thresholds.svg)
- [Stage 1 review](../backend/reports/paysim_three_stage_20260928_172539/stage1_review/SUMMARY.md)
- [Stage 2 review](../backend/reports/paysim_three_stage_20260928_172539/stage2_review/SUMMARY.md)
- [Reset audit](../backend/runtime/validation/paysim_three_stage_20260928_172539_percent_grid_reset.json)
- [Preservation preflight](../backend/runtime/validation/percent_grid_preflight.json)
- [Verification record](../backend/runtime/validation/percent_grid_verification.json)
- [Execution log](../backend/runtime/validation/percent_grid_stage3_20260929_040556.stderr.log)
- [Completion record](../backend/runtime/validation/percent_grid_stage3_20260929_040556.completion.json)
