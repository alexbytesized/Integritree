# Ordinary-SMOTE ratio automation: implementation and continuation plan

## Verified completion update — 2026-11-16 (Asia/Manila)

**The experiment is complete and independently verified. Do not rerun training
to finish this plan.** The older handoff below is preserved as historical context;
its pending steps and incomplete checklist are superseded by this update.

Completed run: `outputs/ordinary_smote_ratios_20261116_retry1/`.
The run finished on 2026-11-16 at 14:51:33 Asia/Manila. It contains exactly three
new full-data fits. The initial failed run remains preserved and performed zero fits.

| Model | F1 at the same 43% cutoff | Fraud detected | False alarms |
|---|---:|---:|---:|
| RF | 0.448798 | 252 | 49 |
| RF-SMOTE 1:100 reference | 0.415990 | 333 | 446 |
| RF-SMOTE 1:200 | 0.435036 | 298 | 250 |
| RF-SMOTE 1:300 | 0.440758 | 279 | 165 |
| RF-SMOTE 1:500 | 0.443493 | 259 | 87 |

| Candidate ratio | Shared selected cutoff | RF F1 | Candidate F1 |
|---|---:|---:|---:|
| 1:200 | 41% | 0.452697 | 0.435658 |
| 1:300 | 41% | 0.452697 | 0.436923 |
| 1:500 | 40% | 0.450704 | 0.450042 |

Reducing oversampling improved RF-SMOTE F1 at the fixed cutoff, with fewer false
alarms but also fewer fraud detections. The 1:500 candidate came closest to RF.
At their shared 40% cutoff it detects 268 fraud cases versus RF's 256, with 101
false alarms versus RF's 58. RF still has slightly higher F1; the numerical gap is
0.00066224. Do not compare the candidate at 40% against RF at 43% to claim a win.
These inspected validation results do not establish statistical equivalence or
generalization performance.

The independent verifier was rerun successfully during this review. It confirmed
all 636,262 validation identities/labels, full score replay for all three saved
candidate models, original reference confusion counts, exact shared-cutoff
selection, deterministic sampling, 458 SHAP reconstructions, and unchanged
input/output/implementation fingerprints. The completed run's offline browser QA
passed for desktop/mobile and all three candidate reports; its desktop report and
paired 1:500 explanation screenshot were visually inspected during this review.
No deployment or test-set evaluation occurred.

- [Completed offline report](outputs/ordinary_smote_ratios_20261116_retry1/report.html)
- [Generated summary](outputs/ordinary_smote_ratios_20261116_retry1/SUMMARY.md)
- [Provenance manifest](outputs/ordinary_smote_ratios_20261116_retry1/manifest.json)
- [Detailed independent results review](RESULTS_REVIEW.md), including subgroup,
  CSV/Parquet, rendered-table and shared-background reconciliation.

The remaining action is researcher discussion of the results. Any additional
training experiment or application change requires a separately agreed scope.

---

**Historical handoff snapshot:** 2026-11-16, 13:29 Asia/Manila (UTC+08:00).

## 1. Historical status before the successful retry

The automation is implemented under `backend/smote_ratio_experiment/`.
**The real three-ratio experiment is not complete. No new full-data forest has
been fitted.** The first attempt stopped at the memory preflight, before training.

A memory-release improvement has since been implemented and passed the complete
12-test experiment suite. It has **not yet been retried against the full dataset**.
There are no experiment or test Python processes running at this handoff.

The next model should continue from this state, not rebuild the experiment or
report synthetic test results as findings about the real dataset.

## 2. User objective and authorization

The researcher wants to investigate whether **less aggressive ordinary SMOTE**
reduces false alarms and improves validation F1, while preserving useful fraud
detections. F1 is the primary discussion metric; success is a verified comparison,
not a guaranteed RF-SMOTE victory.

The user explicitly approved this question:

> For the automated comparison, may I use three lower fraud-to-legitimate
> ratios—1:200, 1:300, and 1:500—and select one shared RF/RF-SMOTE cutoff for each
> comparison by maximizing their mean validation F1? The existing 1:100 model will
> remain the reference.

Their answer was:

> Yes—build and run these three comparisons.

The latest request is to preserve this update/implementation plan in Markdown so
another model can continue if the current session reaches its usage limit.

### Required boundaries

- Keep all new work in `backend/smote_ratio_experiment/`.
- Preserve the existing RF, RF-SMOTE, prepared data, application/API behavior,
  active selection and earlier investigation.
- Use ordinary SMOTE. The separate categorical-sampler experiment was rejected and
  removed at the user's request; do not restore it.
- Exactly three full-data candidate fits: 1:200, 1:300, 1:500. No additional ratios,
  seed search, forest search or automatic deployment.
- Reuse original training/validation partitions and saved preprocessing.
- Train on the training partition only. No validation resampling or test evaluation.
- Five SMOTE neighbors; sampling/model seeds 42; 100 trees; maximum depth 10;
  minimum leaf size 1; all remaining forest parameters match RF; one worker.
- Numerical hour/day treatment remains unchanged. Ordinary SMOTE fractions are
  preserved and audited by the existing backend implementation.
- RF and a candidate must use **the same cutoff within each comparison**.

## 3. Earlier investigation findings

Source: `backend/investigation/outputs/validation_investigation_20261115/`.
The original shared cutoff is 43%, with 636,262 validation transactions,
including 822 fraud cases and 635,440 legitimate cases.

| Model | TP | FP | FN | TN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| RF | 252 | 49 | 570 | 635,391 | 0.837209 | 0.306569 | 0.448798 |
| RF-SMOTE 1:100 | 333 | 446 | 489 | 634,994 | 0.427471 | 0.405109 | 0.415990 |

RF-SMOTE detected 81 additional fraud cases but raised 397 additional false alarms.
The 100,000–<1,000,000 amount band contained 230 of those additional false alarms.
Additional false alarms were in CASH_OUT (205) and TRANSFER (192). These counts
motivate testing sampling intensity; they do not prove causation.

The original training partition has 5,090,096 rows: 5,083,526 legitimate and 6,570
fraud. Expected final fraud counts and synthetic additions for this experiment:

| Ratio | Final training fraud count | Synthetic additions |
|---|---:|---:|
| 1:200 | 25,417 | 18,847 |
| 1:300 | 16,945 | 10,375 |
| 1:500 | 10,167 | 3,597 |

These are expected arithmetic targets, **not completed training results**.

## 4. Implemented design

### Input configuration

`config.yaml` points to these existing, unchanged inputs:

- Models: `backend/artifacts/paysim_three_stage_20260928_172539`
- Prepared data: `backend/data/prepared/paysim_phase2_20260918`
- Validation report:
  `backend/reports/paysim_three_stage_20260928_172539/final_validation`

Relative paths resolve against the configuration file's directory. The runner
supports `--config` and `--run-id`, defaults to its adjacent configuration, and
rejects existing output names.

### Cutoff and reporting protocol

1. For each candidate, evaluate thresholds 1%–100% equally for RF and the candidate.
2. Maximize their **mean validation F1**, using the backend's exact rational
   comparison. Exact ties prefer the cutoff closest to 50%, then the higher cutoff.
3. Apply that same cutoff to RF, the candidate and the existing 1:100 reference.
   The 1:100 reference does not influence the candidate's cutoff selection.
4. Also compare all five models at the original shared 43% cutoff.
5. Keep both views: 43% isolates the sampling change; selected-cutoff results also
   include the effect of threshold selection.

Different ratio comparisons can select different common cutoffs. Do not silently
replace this approved rule with separate RF/candidate thresholds or a pooled
five-model threshold objective.

Reports include confusion counts, precision, recall, F1, MCC, Average Precision,
false alarms per 10,000 legitimate records, undefined reasons and sparse flags.
They include all eight paired outcomes and errors by type, amount, simulation
hour/day, zero amount and merchant indicators, plus type-by-amount/type-by-hour.

Up to five validation cases per candidate-versus-RF outcome/type group are sampled
with seed 42 and stable row ordering. Each case is explained with both models and
the same 200 original-training background records. SHAP probability parity and
additive reconstruction are checked. These are sampled case studies, not global
importance or causal findings.

### Files and responsibilities

| Path inside this folder | Purpose |
|---|---|
| `run.py` | CLI entry point; local plotting cache; no bytecode outside the folder |
| `config.yaml` | Fixed approved ratios, seed, input paths, sampling/support settings |
| `ratio_experiment/core.py` | Protocol validation, matched predictions, metrics, isolated model loader |
| `ratio_experiment/runner.py` | Provenance, training, reloads, evaluations, explanations, failure/completion state |
| `ratio_experiment/reporting.py` | Offline HTML, Markdown, charts and combined comparison |
| `tests/test_protocol.py` | Protocol, ties, identities, groups, support, failure and overwrite checks |
| `tests/test_workflow.py` | Real three-fit workflow on tiny synthetic data, including SHAP/report generation |
| `tests/verify_saved_run.py` | Independent acceptance of the real completed run |
| `tests/preview_report.cjs` | Offline browser, desktop/mobile and explanation-image QA |
| `README.md` | Setup, commands, protocol, interpretation and artifact documentation |
| `outputs/` | Separate immutable run directories, including preserved failures |
| `runtime/` | Temporary tests, plotting/SHAP caches and browser QA |

The runner imports existing backend components and read-only helpers from
`backend/investigation/diagnostics/`. No shared code was changed.
New fitted models use kind `isolated_smote_ratio` and a local loader; ordinary
application bundle loading rejects this format.

## 5. Completed verification

- **12 experiment tests passed** after the latest memory-release change.
- **83 relevant backend regression tests passed** before that small local change:
  training, preprocessing, evaluation, staged selection and SHAP adapter.
- Lint checks for undefined/unused names and formatting passed, including after
  the latest memory-release change.
- Tiny-fixture browser QA passed offline, with no desktop/mobile overflow and
  loaded paired SHAP charts for all three candidate reports.
- The tiny-fixture desktop report was visually inspected. This caught undefined
  values displayed as Python `None`; the local report now renders `undefined`.
- A separate Markdown formatting bug for undefined precision was fixed and the
  entire experiment suite rerun successfully.
- The first real attempt verified source provenance and **replayed both reference
  models' scores across every validation record** before reaching the memory guard.
- No tracked repository files were modified. All implementation changes are in the
  new experiment folder.

The test warnings are three upstream SHAP color-map pending-deprecation warnings,
not failed checks. Synthetic test fits do not count as full-data candidate fits.

## 6. Memory failure and latest fix

Preserved failed run:

`outputs/ordinary_smote_ratios_20261116/`

Its manifest records:

```text
status: failed
new_fits_started: 0
completed_ratios: []
Insufficient available RAM: 2.97 GiB;
estimated working arrays need 2.99 GiB before additional tree/runtime memory.
```

The first run exited cleanly. No full-data model was trained or saved.
Do not delete, overwrite, resume in place, or relabel this failed directory.

The latest code reuses the backend's `release_candidate_memory()`:

```python
gc.collect()
pyarrow.default_memory_pool().release_unused()
```

It releases unused report/Arrow buffers before applying the **unchanged** memory
preflight, after discarding the temporary training DataFrame, and between candidates.
This changes allocation lifetime only. It does not reduce the estimated requirement,
sample the training data, change precision, change model settings or bypass the guard.

The complete synthetic workflow passes after this change. Whether it frees enough
RAM on the real workload is still unverified. Do not claim the memory issue is fixed
until a real retry passes the preflight. Do not close the user's unrelated apps.

## 7. Continuation steps

### Step A — confirm the handoff state

Work from:

```text
C:\Users\Carpicorn\Downloads\Thesis\Tool\Integritree
```

Read this file, `README.md`, and the existing failed run manifest/log. Check that no
newer real run or training process has started since this handoff. Inspect actual
manifests before deciding how many fits remain; do not infer from directory names.

Expected current Git status:

```text
?? backend/investigation/
?? backend/smote_ratio_experiment/
?? test_features_with_isFraud.csv
```

The investigation folder and `test_features_with_isFraud.csv` were already present.
Do not delete or commit them as part of this work. No commit was requested.

### Step B — retry the full experiment with a new run name

The user already authorized the three comparisons. The first attempt performed
zero fits, so a reviewed retry does not add an extra candidate model.

```powershell
& .\backend\.venv\Scripts\python.exe -B .\backend\smote_ratio_experiment\run.py --run-id ordinary_smote_ratios_20261116_retry1
```

Use another unused name if this one has since been created. Monitor the log and
manifest. Training is sequential and may take tens of minutes. Communicate meaningful
progress to the user while the run is active.

If RAM remains insufficient, preserve that failed attempt and diagnose allocation
lifetime or actual available memory. Do not lower the guard or change the approved
scientific settings just to get past it. If user action is necessary to free RAM,
explain the measured requirement clearly.

If a retry fails **after fitting any candidate**, do not blindly rerun all three.
Inspect which model artifacts and stages completed. Preserve fitted models and
design a verified continuation that avoids unnecessary new fits, or explain the
remaining issue to the user. The current CLI does not implement resume.

Do not edit experiment implementation files while a real run is active: it hashes
them at startup and checks that they remain unchanged at completion.

### Step C — independent acceptance after completion

```powershell
& .\backend\.venv\Scripts\python.exe -B .\backend\smote_ratio_experiment\tests\verify_saved_run.py .\backend\smote_ratio_experiment\outputs\ordinary_smote_ratios_20261116_retry1
```

This must verify input/output/code hashes, all 636,262 validation identities and
labels, full reloaded-model score replay for each candidate, original reference
confusion counts, the exact shared-cutoff optimum via an independent brute-force
calculation, deterministic case sampling and every exported SHAP reconstruction.

Then run browser QA:

```powershell
node .\backend\smote_ratio_experiment\tests\preview_report.cjs .\backend\smote_ratio_experiment\outputs\ordinary_smote_ratios_20261116_retry1
```

Visually inspect `runtime/qa/<run-id>/desktop.png`, `mobile.png`, and at least one
`paired_<denominator>.png`. Confirm the actual-data report is readable and links are
local. Do not rely exclusively on the earlier synthetic report's visual check.

### Step D — finish the handoff and report the actual outcome

Update this file with the completed run name, selected cutoffs, key metrics, final
verification results and any limitations. Provide the user links to the offline
report, Markdown summary and README.

Explain separately:

1. Whether a lower ratio improved F1 at the unchanged 43% cutoff.
2. What changed when each pair used its selected common cutoff.
3. The precision/recall and false-alarm tradeoffs, with counts and supported patterns.

Retain the exploratory validation caveat: these data have already been inspected
and used for selection. Do not present the best result as unbiased test performance,
claim significance without analysis, or automatically select a deployed model.

## 8. Completion checklist

- [x] Approved protocol and isolated code/configuration implemented.
- [x] Tests for protocol, training/reload, sampling, SHAP and reporting pass.
- [x] Relevant shared backend regression checks pass.
- [x] Synthetic reports pass offline desktop/mobile QA.
- [x] First real attempt preserved; no candidate fit occurred.
- [x] Release unused report buffers without changing the memory guard; tests pass.
- [x] Real retry passes memory preflight.
- [x] Exactly three approved full-data candidate fits complete.
- [x] All shared-cutoff and fixed-43% comparisons, explanations and reports complete.
- [x] Independent full-data acceptance verifier passes.
- [x] Actual-data reports pass browser and visual QA.
- [x] Source preservation and isolation confirmed at completion.
- [x] Actual results and final artifact links delivered to the researcher.

**Completed run:** `ordinary_smote_ratios_20261116_retry1`. All three fits, verifications,
explanations and reports finished successfully. Selected cutoffs: 1:200 → 41%, 1:300 → 41%,
1:500 → 40%. At 43%, all lower ratios improved F1 over the 1:100 reference (best: 1:500 at
0.4435 vs 1:100 at 0.4160). RF alone remains highest F1 (0.4488). See the completed run
artifacts for full metrics, subgroup breakdowns, and SHAP case studies.
