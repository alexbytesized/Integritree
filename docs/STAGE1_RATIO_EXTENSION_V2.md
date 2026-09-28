# Stage 1 ratio extension: protocol revision 2

Authorized 2026-09-28 after reviewing the completed original Stage 1. The original
study selected 1:10, with RF-SMOTE validation AP 0.3674764766459544. Its frozen
decision and every historical artifact remain unchanged. This amendment is
data-informed development on the same validation set, not independent confirmation.

## Bounded candidate set and decision

The complete amended set is **1:1, 1:2, 1:3, 1:5, 1:10, 1:20, 1:50, 1:100**.
Only the final three are new fits. The selector checks/reuses the five original
candidates first, then fits the additional RF-SMOTE models with verified benchmark
RF reuse. Candidate order does not determine the winner.

All eight use 100 trees, maximum depth 10, and minimum leaf size 1. Training data,
preprocessing, features, seeds, ordinary SMOTE with k=5, validation records, and
all other forest settings remain unchanged. Float sampling strategies for the new
ratios are .05, .02, and .01, respectively. The benchmark remains unresampled.

Select the highest RF-SMOTE validation Average Precision across **all eight**;
exact ties prefer the smaller minority/majority ratio. The .50 classification
metrics are supplementary and do not select the ratio. No further expansion is
included regardless of the results; RF-SMOTE is not required to outperform RF.

The amended winner is the ratio to carry into Stage 2 if continuation is separately
authorized. Until revision 2 completes, no amended winner exists. The original
1:10 decision remains valid for the original study, not silently replaced in it.
The current authorization ends after Stage 1: no forest tuning, threshold search,
SHAP job, or held-out test evaluation.

## Versioning and provenance

Configuration: `backend/configs/validation_three_stage_v2.yaml`.
Research revision is 2; schema version remains 1 because the selection algorithm
and structure are compatible. Revision-1 plans without explicit revision fields
remain readable and resumable. Resume never changes a frozen plan in place.

New run: `backend/artifacts/paysim_stage1_ratio_extension_v2_20260928/`.
Original run: `backend/artifacts/paysim_three_stage_validation_20260927/`.
Original Stage 1 decision SHA-256:
`601a52cd43d826b036e1efc3b722be88e704e69365f285f4625668c3c2fe27e1`.

The new frozen plan includes the complete candidate list, revision, amendment
disclosure, dependency versions, implementation hashes, and reuse references.
Matching data provenance, forest settings, seeds, dependencies, and file hashes
are required for reuse. Validation identities, labels, and benchmark scores must
agree across all candidates. Training remains one-worker with RAM checks.

## Execution from `backend/`

Start once, after the backend suite passes:

```powershell
& ./.venv/Scripts/python.exe scripts/select_three_stage.py --prepared data/prepared/paysim_phase2_20260918 --config configs/experiment.yaml --protocol configs/validation_three_stage_v2.yaml --run-id paysim_stage1_ratio_extension_v2_20260928 --jobs 1 --stop-after-stage 1
```

After an interruption, resume the **new** run, not the original:

```powershell
& ./.venv/Scripts/python.exe scripts/select_three_stage.py --prepared data/prepared/paysim_phase2_20260918 --config configs/experiment.yaml --protocol configs/validation_three_stage_v2.yaml --resume artifacts/paysim_stage1_ratio_extension_v2_20260928 --jobs 1 --stop-after-stage 1
```

The final Stage 1 decision/table are under the new run's `stages/01_smote_ratio/`.
Its readable, fingerprinted report is
`backend/reports/paysim_stage1_ratio_extension_v2_20260928/stage1_review/`.
Completion status is `awaiting_next_stage`, not a completed final model selection.

## Manuscript disclosure

Present the original five-ratio study and this extension as separate protocol
versions. Explain the boundary optimum and observed AP trend that motivated the
extension; report all eight candidates, not just improvements. Acknowledge repeated
validation use, possible selection optimism, one seed set, and ratio/forest
interactions that sequential selection may miss. Do not claim a global optimum or
general RF-SMOTE superiority. The untouched test remains reserved for a separately
authorized final evaluation after configuration and threshold selection.
