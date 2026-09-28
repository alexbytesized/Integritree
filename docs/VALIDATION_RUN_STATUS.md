# Validation run status

## Reset record

On 2026-09-28 the researchers requested a clean restart: prior trained models,
validation results, and explanation caches were permanently removed. Raw data,
prepared partitions, preparation verification, and Word reference documents were
retained. This reset does not make the protocol historically preregistered or
remove the limitation of prior inspection of the same validation set.

Cleanup removed 37 generated run/output entries. Preservation hashes are recorded
in `backend/runtime/validation/reset_verification.json`. All 16 retained files
passed unchanged SHA-256 checks after cleanup, including both Word documents.
Local Markdown links and Git whitespace checks passed.
The revised suite passed **169 tests** with five upstream deprecation warnings;
`pip check` found no broken requirements.

## Superseded five-ratio attempt

Run: `paysim_three_stage_20260928_084238`.
Started 2026-09-28 at 08:42 Manila time. **Stopped at RAM preflight**, before
any model was fitted: 2.20 GiB available versus 3.15 GiB estimated working
arrays for the first 1:10 candidate, excluding additional model/runtime memory.
Metadata records `status: failed`, `MemoryError`; no ratio is selected.
The frozen plan compares the original five ratios with 100 trees/depth 10/leaf 1,
one worker, and `--stop-after-stage 1`. The frozen reuse inventory has zero models and zero reports.

Live evidence:

- `backend/artifacts/paysim_three_stage_20260928_084238/metadata.json`
- `backend/artifacts/paysim_three_stage_20260928_084238/search_plan.json`
- `backend/runtime/validation/paysim_three_stage_20260928_084238.stderr.log`
- `backend/runtime/validation/paysim_three_stage_20260928_084238.stdout.log`

Stage 2, Stage 3, final test evaluation, and SHAP are outside this execution.
No result is inferred before the frozen Stage 1 decision and review are verified.

## Current eight-ratio protocol

Stage 1 now evaluates 1:100, 1:50, 1:20, 1:10, 1:5, 1:3, 1:2, and 1:1 in that
order. Reference forest, objectives, seeds, and later-stage grids are unchanged.
The failed five-ratio plan above remains intact as a superseded attempt; do not
resume it with the updated protocol. A fresh uniquely named run is required.

The expanded suite passed **172 tests** with five upstream deprecation warnings;
`pip check` found no broken requirements. Tests cover all eight resampling counts,
exact AP ties, eight-candidate Stage 1 completion, benchmark reuse, memory release,
held-out guards, and rejection of protocol changes on resume. Retained data, Word
documents, and all files of the superseded attempt passed unchanged-hash checks.

## Completed Stage 1

Run: `paysim_three_stage_20260928_172539`.
Stage 1 ran from **17:25 to 18:59 Manila time on 2026-09-28** and froze
**1:100**. All eight candidates completed. RF-SMOTE AP was **0.38119617175331205**
for the selected ratio; benchmark RF AP was **0.3915279330994021**.
The frozen plan, decision, candidate table, and exported review passed integrity
checks. No threshold or final model configuration was selected by this stage.

Review: `backend/reports/paysim_three_stage_20260928_172539/stage1_review/`.
The original review remains an unchanged snapshot of Stage 1 completion; its
selection-metadata hash refers to that export time, before later execution history.
The scientific search plan remains unchanged, SHA-256:
`1df14b38aa3fa5305649f228b4740d10e1e74c870e01b254d0f9b6ea98a89b25`.

## Stage 2 continuation

Continue the same run at 1:100 across trees {100, 200}, depths {10, 20}, and
minimum leaf sizes {1, 10, 50}. Reuse the completed reference pair and train
11 further pairs. Select the highest mean validation AP; exact ties prefer
shallower depth, fewer trees, then larger leaves.

The full backend suite passed **181 tests** with five upstream deprecation warnings;
`pip check` found no broken requirements. A final control-lock refinement passed
all **9 targeted pause tests**. Paused/resumed and uninterrupted synthetic runs
produced identical scores, candidate metrics, and the selected forest. Requests
during the last candidate correctly completed Stage 2 rather than pausing.

Stage 2 resumed at **22:35 Manila time on 2026-09-28**, worker PID
**22300**, with `--stop-after-stage 2`. The initial observation confirmed:

- All Stage 1 evidence reverified; frozen plan, decision/table, and original review unchanged.
- The 100-tree/depth-10/leaf-1 reference pair reused: **1 of 12 Stage 2 candidates complete**.
- The first new pair, **100 trees/depth 10/leaf 10**, is training its benchmark RF
  on 5,090,096 original rows. No new forest winner exists yet.
- Candidate RAM preflight passed: **6.25 GiB available**, versus **3.00 GiB**
  estimated working arrays, excluding additional model/runtime memory.

This is a dated running-status observation. Live evidence:

- `backend/artifacts/paysim_three_stage_20260928_172539/metadata.json`
- `backend/artifacts/paysim_three_stage_20260928_172539/search_progress.json`
- `backend/runtime/validation/stage2_20260928_223521.stderr.log`
- `backend/runtime/validation/stage2_20260928_223521.stdout.log`

Stage 3, official test evaluation, and SHAP remain outside this execution.

## Pause and resume

Request a pause from `backend/` while the selection worker is running:

```powershell
& ./.venv/Scripts/python.exe scripts/pause_selection.py --run artifacts/paysim_three_stage_20260928_172539
```

`pause_requested` is an acknowledgement, not a stopped worker. Wait for `paused`
in metadata and worker exit: the current pair and validation finish first.
Resume the same run after the worker exits:

```powershell
& ./.venv/Scripts/python.exe scripts/select_three_stage.py --prepared data/prepared/paysim_phase2_20260918 --config configs/experiment.yaml --protocol configs/validation_three_stage.yaml --resume artifacts/paysim_three_stage_20260928_172539 --jobs 1 --stop-after-stage 2
```

Successful Stage 2 ends with `awaiting_next_stage`, `completed_stage: 2`, and
`stage: forest_selected`. Its review is exported to
`backend/reports/paysim_three_stage_20260928_172539/stage2_review/`.
Check live `metadata.json`, `search_progress.json`, and the current execution logs
for updates. Do not resume the superseded five-ratio attempt.
