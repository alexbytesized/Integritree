# Full validation run started 2026-09-24

## Revision-2 Stage 1 extension started 2026-09-28

The original Stage 1 completed at **23:29 Manila time on 2026-09-27**, selecting
1:10 (RF-SMOTE AP 0.3674764766459544). That study is no longer running; its
metadata, frozen decision, reports, and model artifacts remain unchanged.

The researchers authorized the [bounded revision-2 extension](STAGE1_RATIO_EXTENSION_V2.md)
after reviewing those results: add 1:20, 1:50, and 1:100, compare all eight ratios
by the unchanged RF-SMOTE AP rule, and stop after Stage 1. This is a new experiment,
not a modification of the original frozen plan. All **172 backend tests passed**
(five upstream deprecation warnings); `pip check` reported no broken requirements.
Tests cover original-artifact preservation, reuse of the original five models and
reports, the new resampling counts, and backward compatibility with revision-1 plans.

The new run **paysim_stage1_ratio_extension_v2_20260928** started at **00:53 Manila
time**, worker PID **24696**. At the initial running-status check, all five original
candidates had been verified and reused without new fits. The new **1:20 RF-SMOTE**
model was training on **5,337,702 rows**, with the benchmark RF reused and verified.
Reference forest remains 100 trees/depth 10/leaf 1, one worker. Memory checks remain
enabled. No amended winner exists yet at this observation.

Live evidence:

- `backend/artifacts/paysim_stage1_ratio_extension_v2_20260928/metadata.json`
- `backend/artifacts/paysim_stage1_ratio_extension_v2_20260928/search_progress.json`
- `backend/runtime/validation/20260928_005315_stage1_v2.stderr.log`
- `backend/runtime/validation/20260928_005315_stage1_v2.stdout.log`

Frozen amended plan SHA-256:
`cb33676c8905d717287992ad376c2bd6a65e49268b39fcae5d8034c1a00dd59d`.
Read the live evidence for completion or failure; this entry records the initial
running state. The process stops before Stage 2, threshold selection, or test use.
On completion, its review is under
`backend/reports/paysim_stage1_ratio_extension_v2_20260928/stage1_review/`.
Use the new protocol and new run ID in the amendment's resume command if interrupted.

Earlier entries below are historical observations.

## Stage 1-only execution resumed 2026-09-27

The researchers authorized **Stage 1 only**. The selector now supports
`--stop-after-stage 1`; a regression test forbids entry to forest and threshold
selection. All **170 backend tests passed**, with five upstream deprecation
warnings; dependency checks passed.

The saved run resumed at **22:45 Manila time** (14:45 UTC), worker PID **24944**.
Its execution history records the Stage 1 boundary and current selector hash;
the frozen research plan and its SHA-256 remain unchanged. The first 1:10
candidate passed preflight with 5,687,021,568 available bytes versus 3,379,235,264
estimated working-array bytes. It reused the verified 100-tree/depth-10/leaf-1
benchmark RF and began fitting RF-SMOTE on 5,591,878 training rows.

This is a running-status observation, not a completed Stage 1 result. Check live
metadata and logs for completion or failure. No ratio is frozen yet at this
observation; Stage 2, Stage 3, and test evaluation are outside this authorization.
The process will stop after freezing Stage 1, with `status: awaiting_next_stage`.

Live evidence:

- `backend/artifacts/paysim_three_stage_validation_20260927/metadata.json`
- `backend/artifacts/paysim_three_stage_validation_20260927/search_progress.json`
  (created when the first candidate finishes validation)
- `backend/runtime/validation/20260927_224553_stage1.stderr.log`
- `backend/runtime/validation/20260927_224553_stage1.stdout.log`

On success, candidate metrics and the frozen ratio decision are under
`stages/01_smote_ratio/` in that artifact run; its readable, fingerprinted review
is under `backend/reports/paysim_three_stage_validation_20260927/stage1_review/`.
Supplementary classification metrics use .50; no final fraud threshold is selected.
If interrupted, resume **Stage 1 only** from `backend/`:

```powershell
& ./.venv/Scripts/python.exe scripts/select_three_stage.py --prepared data/prepared/paysim_phase2_20260918 --config configs/experiment.yaml --protocol configs/validation_three_stage.yaml --resume artifacts/paysim_three_stage_validation_20260927 --jobs 1 --stop-after-stage 1
```

The earlier failed attempt below is retained as history, not the current status.

## Three-stage revision authorized 2026-09-27

The new [protocol](THREE_STAGE_VALIDATION.md) adds five-ratio AP selection,
12 paired forests including minimum leaf size, and an exact all-score threshold
search. Its separate schema-2 selector, stage checkpoints, verified reuse, and
report export are implemented. Verification passed **169 backend tests** with
five upstream deprecation warnings; `pip check` found no broken requirements.
Synthetic checks cover all ratios, the 12-setting grid, exact threshold selection
(including cutoffs above .95 and ties), matching leaf settings, reuse, corruption
rejection, and recovery after an interruption following a completed fit.

The full-data run `paysim_three_stage_validation_20260927` was attempted at
18:15 Manila time (10:15 UTC). Its reuse inventory verified five existing model
bundles and recorded seven validation reports for candidate-specific checks.
The first candidate, 1:10 at 100 trees/depth 10/leaf 1, stopped at RAM preflight:
**3.10 GiB available versus 3.15 GiB estimated working arrays**, excluding
additional tree/runtime memory. No candidate was newly fitted and no stage froze.
At that observation the run was inactive, with `status: failed`, `MemoryError`.

Implementation and synthetic verification are complete; full-data execution and
verification of the final frozen result were blocked by available RAM. Free
memory before resuming; later ratios may require larger working arrays (up to
about 4.65 GiB for 1:1, plus trees/runtime). The guard was not reduced or bypassed,
and unrelated applications were not closed. Historical artifacts and the held-out
test remain untouched. Resume from `backend/` with:

```powershell
& ./.venv/Scripts/python.exe scripts/select_three_stage.py --prepared data/prepared/paysim_phase2_20260918 --config configs/experiment.yaml --protocol configs/validation_three_stage.yaml --resume artifacts/paysim_three_stage_validation_20260927 --jobs 1
```

Evidence: `backend/artifacts/paysim_three_stage_validation_20260927/metadata.json`
and `search_plan.json`. The plan SHA-256 is
`283346c83dde74ef26fbee23adef8f0218cbd2abe83331f29c69d5e4ee3ae0bb`.
No new ratio, forest configuration, common cutoff, or final validation report
exists yet for this run. Do not populate new manuscript results from it.

## 1:3 ratio sensitivity study approved 2026-09-25

The completed 1:1 artifacts and validation reports remain immutable. A separate
validation-only configuration now defines ordinary SMOTE at one fraud row per
three legitimate rows after resampling, with 200 trees and depth 20. That forest
setting was inherited from the 1:1 validation search; the ratio was proposed only
after its results were reviewed.

The implementation supports verified reuse of the existing 200/depth-20 benchmark
RF, so only the 1:3 RF-SMOTE forest needs a new fit. The comparison uses the same
636,262 validation identities and the common .05-.95 threshold grid. It is
descriptive: no ratio, cutoff, or final model is selected. The held-out test split
must not be read.

The full job subsequently completed. It generated 1,687,938 fraud rows, producing
5,083,526 legitimate and 1,694,508 fraud training rows (6,778,034 total). The
existing benchmark RF was reused with matching source and copied-model hashes.
The new artifact and reload verification are complete.

The validation comparison passed identity, label, and exact benchmark-score checks.
PR-AUC was 0.362801 for benchmark RF, 0.335711 for RF-SMOTE 1:1, and 0.330708 for
RF-SMOTE 1:3. At the .95 reference checkpoint, 1:3 improved precision from 0.143695
to 0.293948, F1 from 0.213774 to 0.328502, and MCC from 0.243280 to 0.329825 versus
1:1, while recall decreased from 0.417275 to 0.372263. These findings do not select
a ratio or cutoff. Evidence is under
`backend/reports/paysim_smote_ratio_comparison_validation_20260925/`.
The complete backend suite passed 150 tests with five upstream deprecation warnings,
and the installed dependency check reported no broken requirements.

The researchers authorized execution of the approved four-configuration search.
The run started at 02:54 Manila time on 2026-09-24 (18:54 UTC on 2026-09-23).

Run directory: `backend/artifacts/paysim_validation_selection/`.
The original `paysim_phase3_baseline_20260920` bundle is reused unchanged.
All new candidates train with one worker, the original training split, matching
RF settings, and ordinary 1:1 SMOTE. Selection uses validation only. No test
evaluation or SHAP job is part of this execution.

At the resumed status update (03:56 Manila, 2026-09-24), both 100-tree candidates
have completed validation. The 200-tree/depth-10 candidate is training its RF
model (worker PID 26088). No winning configuration or cutoff exists yet.
This is a dated observation; inspect the live metadata/logs for current status.

## Memory stop and resume

The initial search stopped before training the 200-tree/depth-10 candidate:
available RAM was 3.73 GiB, below the 4.65 GiB working-array estimate. This
estimate excludes additional tree/runtime memory. Both completed candidates
were retained. The original failure metadata and logs are archived under
`backend/reports/paysim_validation_memory_stop_20260924/`.

The selection loop now releases the previous candidate's prediction table and
unused Arrow allocations before the next training preflight. The targeted
Phase 4 workflow suite passed all six tests, including a regression check that
previous report frames are released before the next training call. The fix
does not change the dataset, SMOTE, RF settings, or memory safety check.

The same search resumed at 03:49 Manila. Its new candidate preflight recorded
5,062,938,624 available bytes against 4,989,696,512 estimated array bytes, so
the check passed with limited headroom. The two completed candidates were
reused. The automatic review and desktop alerts were reattached to PID 26088.

Live evidence:

- `backend/artifacts/paysim_validation_selection/metadata.json`: overall status.
- `backend/artifacts/paysim_validation_selection/search_progress.json`: written
  after each complete candidate evaluation; absent before the first finishes.
- `backend/runtime/exports/validation-selection-resume-20260924.log`: active
  training/selection log; the original log is `validation-selection-20260924.log`.
- `backend/runtime/exports/validation-review-20260924.log`: automatic review log.

After successful selection, the queued local review helper verifies the frozen
selection, independently checks F1 using scikit-learn, exports candidate and
threshold tables, and evaluates the selected pair on **validation** at the frozen
common threshold. Outputs will be under:

`backend/reports/paysim_validation_selection_review_20260924/`

Expected files after that review completes:

- `SUMMARY.md`: readable selected settings, candidate F1, and final validation metrics.
- `candidate_scores.csv`: both models' metrics for each candidate at cutoff 0.50.
- `threshold_scores.csv`: both F1 scores and their mean for each common cutoff.
- `evaluation/selected_threshold_validation/`: paired predictions, metrics,
  statistical results, figures, and provenance at the selected cutoff.
- `metadata.json`: review status and file fingerprints.

These outputs must not be described as final held-out test findings. All
candidate bundles and validation reports remain available for inspection.
The local review helper is `backend/runtime/exports/review_validation_20260924.py`.
It waits for the training process to exit and refuses to infer a winner if the
selection is incomplete. If the search fails, resume the original run using the
README command; rerun the review helper without `--wait-pid` after it completes.
Do not launch a second training process while the current one is active.

Temporary idle-sleep prevention is scoped to the active worker/helper threads;
it is released when they exit. No permanent Windows power setting was changed.

## Desktop milestone alerts

Requested by the researchers and enabled on 2026-09-24. A hidden Windows desktop
notification monitor polls the run metadata every ten seconds and alerts on:

- Each completed candidate evaluation, with its mean validation F1.
- The selected shared RF configuration and common cutoff.
- Completion of the readable validation review and exports.
- Selection failure, unexpected worker exit, or a reported review failure.

The initial desktop alert was confirmed by Windows' `BalloonTipShown` event.
These are local Windows notifications, not scheduled messages in the Codex chat.
They depend on the computer/user session and Windows notification settings.
The monitor exits after completion or failure and does not restart training.

Monitor files are in `backend/runtime/exports/`: `notify_validation_20260924.ps1`,
`validation-notifier-state-20260924.json`, and
`validation-notifications-20260924.jsonl`. The state records the active monitor PID
and its latest check; the event log distinguishes a requested alert from a
confirmed shown event. The resumed desktop monitor PID is 37300; Windows
confirmed its initial alert at 03:56 Manila. The previous monitor exited after
reporting the memory stop. Previously announced completed candidates are not
announced again on resume.
