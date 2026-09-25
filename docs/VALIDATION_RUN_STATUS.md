# Full validation run started 2026-09-24

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
