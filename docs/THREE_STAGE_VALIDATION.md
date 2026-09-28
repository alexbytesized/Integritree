# Three-stage validation protocol

## Decisions, in order

1. Compare ordinary SMOTE fraud:legitimate ratios 1:100, 1:50, 1:20, 1:10, 1:5, 1:3, 1:2, and 1:1
   using 100 trees, maximum depth 10, and minimum leaf size 1. Select the highest
   RF-SMOTE validation Average Precision (AP). Exact ties prefer the smaller ratio.
2. Freeze the ratio. Compare 12 shared forests: trees {100, 200}, maximum depth
   {10, 20}, and minimum leaf size {1, 10, 50}. Select the highest arithmetic mean
   validation AP across benchmark RF and RF-SMOTE. Exact ties prefer shallower
   depth, fewer trees, then larger minimum leaf size.
3. Freeze the fitted pair. Evaluate exactly 100 thresholds: 0.01, 0.02, ..., 1.00
   (1% through 100%; no 0% candidate). Select the highest mean validation F1 across both
   models, breaking exact ties by proximity to 0.50, then higher threshold.
   Prediction is fraud when `score >= threshold`. Thresholds are not rounded.

AP uses Average Precision, not trapezoidal PR-curve area. Classification metrics
at 0.50 supplement Stages 1 and 2 but do not determine their winners. The threshold
sweep retains precision, recall, F1, MCC, accuracy, and confusion counts per model
at every candidate; undefined precision/MCC are blank, not silently zero. Exact
rational mean F1 and integer percentage distances from 50 determine Stage 3. Sorting
and prefix counts avoid repeated training, prediction, or full-record scans per
threshold. The selected cutoff is best among the 100 tested whole percentages.
All 100 candidates appear in the exported tables and plot. No assumption is made that larger leaves or a particular ratio help.

Sequential freezing can miss ratio/forest interactions. Repeated use of one
validation set, one seed set, and prior inspection of validation results must
be disclosed. Validation metrics are development evidence, not unbiased final
generalization estimates or grounds to claim RF-SMOTE superiority.

## Research boundaries

Use the original prepared training, validation, and test partitions. Preprocessing
is unchanged. SMOTE uses training records only, `k_neighbors=5`, and unchanged
seeds. Benchmark RF always uses the original class distribution. Both models use
identical forest settings, including minimum leaf size; all other parameters are
unchanged. There is no resampling of validation, refit on combined partitions,
test evaluation, prediction/API change, or SHAP execution in this workflow.

The output selects a configuration for **both** RF and RF-SMOTE, not one model as
the winner. Stop after freezing the pair and producing its validation report.
Official held-out test evaluation requires separate authorization.

## Reproducibility, reuse, and layout

`backend/configs/validation_three_stage.yaml` declares the versioned search;
`experiment.yaml` supplies unchanged data, features, seeds, and remaining settings.
Completed selected-bundle manifests use schema 2. Two-step selection bundles
are unsupported; ordinary fitted model bundles retain schema 1. Selection bundles reference immutable
candidate models rather than duplicating their large serialized forests.

The selector snapshots eligible artifact locations. Reuse requires matching full
training configuration, preparation provenance, dependency versions, and hashes.
Benchmark RF can be reused across ratios only for identical forest settings; a new
minimum leaf size requires a matching new benchmark. Scores must come from the
matching model, validation split, evaluation policy, and dependencies. Identities
and labels must match across all candidates; benchmark-score hashes must match
across ratios for identical forests. No test-feature file is read by this workflow.

Each run under `backend/artifacts/<run-id>/` contains:

- `search_plan.json`: frozen protocol, base configuration, preparation hash, reuse
  inventory; `search_progress.json`: completed candidate checkpoints.
- `candidates/<ratio-and-full-forest>/`: new paired bundles; matching bundles from this run remain at their verified locations.
- `validation/<ratio-and-full-forest>/`: new validation reports; verified reports may be reused on resume.
- `stages/01_smote_ratio/` and `stages/02_random_forest/`: candidate metrics CSVs
  and frozen decisions with objectives and evidence references.
- `stages/03_threshold/`: exact decision, full-precision CSV and Parquet threshold
  tables, and an SVG curve (plot points may be sampled; tables are complete).
- `selection.json`, `metadata.json`, `SUMMARY.md`: frozen ratio, forest, common
  cutoff, model/report references, fingerprints, and selection rationale.

`backend/reports/<run-id>/` collects the stage tables and figures plus
`final_validation/`, with selected-threshold metrics, paired continuous scores,
precision-recall curves, and confusion figures. Full-precision Parquet files are
authoritative; CSVs use 17 significant digits for float round-tripping.

Stages freeze before the next starts. Resume verifies the original plan and saved
evidence, reuses completed fits/reports even after a checkpoint interruption, and
recomputes decisions to check integrity. Failed attempts remain available. A process
lock prevents simultaneous writers. Training uses one worker and ratio-dependent
RAM preflight; passing it is not a peak-memory guarantee.

## Execution (from `backend/`)

Run software checks before full-data training:

```powershell
& ./.venv/Scripts/python.exe -m pytest -q
& ./.venv/Scripts/python.exe -m pip check
```

The active run completed Stages 1 and 2. Resume its authorized **Stage 3** continuation:

```powershell
& ./.venv/Scripts/python.exe scripts/select_three_stage.py --prepared data/prepared/paysim_phase2_20260918 --config configs/experiment.yaml --protocol configs/validation_three_stage.yaml --resume artifacts/paysim_three_stage_20260928_172539 --jobs 1 --stop-after-stage 3
```

`integritree-select` and `integritree-select-three-stage` invoke the same selector.
For a separate fresh study, omit `--resume` and use an unused or generated run ID;
`--stop-after-stage 1` stops at ratio selection. Ordinary resume never changes a
frozen plan. An explicitly authorized Stage 3 reset uses `scripts/reset_stage3.py`
with `--run artifacts/<run-id>` to amend only the threshold candidate policy, remove
Stage 3 outputs, and regenerate Stage 1-2 review snapshots. A reset audit records
the cleanup and preservation hashes; retry that command after an interruption
before resuming. The current run reuses all fitted candidates without retraining.

Stage 2 ends with `status: awaiting_next_stage`, `completed_stage: 2`, and
`stage: forest_selected`. Its decision and candidate table are under
`stages/02_random_forest/`; its fingerprinted export is under
`reports/<run-id>/stage2_review/`. Both stages' frozen evidence is verified on
resume. A Stage 2-only run cannot be loaded as a final selected model pair.
The authorized `--stop-after-stage 3` continuation selects the threshold and
exports validation results. Official test evaluation and SHAP remain separate.
The protocol was revised after validation inspection; regenerated Stage 1-2
reviews preserve their candidate decisions but are not original report snapshots.

### Graceful candidate pause

From a separate terminal in `backend/`:

```powershell
& ./.venv/Scripts/python.exe scripts/pause_selection.py --run artifacts/paysim_three_stage_20260928_172539
```

This creates a run-local `pause_request.json`; it does not terminate or signal the
worker. Requests are scoped to the current execution and coordinated by a short
control lock. The worker checks between candidates, completing both model fits,
validation, and its progress checkpoint before reporting `status: paused` and
releasing the run lock. Metadata records the last completed candidate and pause
time. Wait for that state and process exit before shutting down. If the last
candidate finishes, the requested stage is frozen normally instead of pausing.

Resume with the intended `--stop-after-stage` value. The acknowledged request is consumed, and
completed models/reports are verified and reused. No parameters, seeds, data, or
model-training algorithms change. Paused/resumed synthetic runs must match
uninterrupted scores and decisions. Partial fits from abrupt interruptions are not
resumable; the unfinished candidate may need retraining. Failure records remain
available. Do not change the plan or disable RAM checks to bypass a failure.

See [current run status](VALIDATION_RUN_STATUS.md).

## Manuscript reporting

Describe the stages, objectives, exact tie rules, candidate ranges, unchanged
partitions, and training-only SMOTE. Report all eight ratio candidates, all 12
forest candidates, the exact common cutoff and candidate count, and each model's
selected-threshold validation metrics. Include sequential-interaction and
post-validation-revision limitations. Report only verified outputs from the current run; do not substitute deleted
results for unexecuted stages.
Populate numerical conclusions only from a completed, verified run. Never label
validation as test results or claim this workflow chooses between models.
