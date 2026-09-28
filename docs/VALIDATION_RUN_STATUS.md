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

## Active eight-ratio Stage 1 run

Run: `paysim_three_stage_20260928_172539`. Started **17:25 Manila time on 2026-09-28**.
At the initial check, status was `running`, worker PID **5408**.
The first candidate (1:100) passed memory preflight and began fitting benchmark
RF on **5,090,096 original training rows**, with 100 trees/depth 10/leaf 1.
All eight preflight estimates passed with about **5.72 GiB available**; estimates
range from **3.00 GiB** at 1:100 to **4.65 GiB** at 1:1, excluding model/runtime
memory. Memory checks remain enabled for each candidate.

The frozen reuse inventory is empty. Benchmark reuse will occur only among
newly fitted candidates within this run. The plan SHA-256 is
`1df14b38aa3fa5305649f228b4740d10e1e74c870e01b254d0f9b6ea98a89b25`.
This is a running-status observation, not a completed result; no ratio is frozen
yet. Check the live evidence for subsequent completion or failure:

- `backend/artifacts/paysim_three_stage_20260928_172539/metadata.json`
- `backend/artifacts/paysim_three_stage_20260928_172539/search_progress.json` (created after the first candidate)
- `backend/runtime/validation/paysim_three_stage_20260928_172539.stderr.log`
- `backend/runtime/validation/paysim_three_stage_20260928_172539.stdout.log`

On completion, the frozen ratio decision and candidate table are under
`stages/01_smote_ratio/`, and its review is under
`backend/reports/paysim_three_stage_20260928_172539/stage1_review/`.
The worker stops at `awaiting_next_stage`; no Stage 2, Stage 3, test evaluation,
or SHAP execution is included.

Resume only after this worker exits, from `backend/`:

```powershell
& ./.venv/Scripts/python.exe scripts/select_three_stage.py --prepared data/prepared/paysim_phase2_20260918 --config configs/experiment.yaml --protocol configs/validation_three_stage.yaml --resume artifacts/paysim_three_stage_20260928_172539 --jobs 1 --stop-after-stage 1
```

Do not resume the superseded five-ratio run with the eight-ratio protocol.
