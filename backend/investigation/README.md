# RF / RF-SMOTE validation investigation

One command produces an offline diagnostic report from the **existing validation
predictions**. All investigation-specific code, configuration, tests, outputs, logs,
and caches are kept in this folder. It references the existing backend environment,
prepared dataset, selected model bundle, and evaluation report in place.

This workflow does not train models, generate synthetic examples, fit probability
calibrators, select deployed settings, or evaluate the test set. F1 is the priority
for discussing a subsequent recalibration experiment; all five study metrics remain
visible. A favorable RF-SMOTE result is not a completion criterion.

## Run

Use the existing Python 3.12 backend environment with the locked dependencies and
editable `integritree-backend` installation described in [the backend README](../README.md).
No additional packages, web server, browser downloads, or network connection are
required by the investigation.

From the repository root in PowerShell:

```powershell
& .\backend\.venv\Scripts\python.exe -B .\backend\investigation\run.py
```

Optional settings and a descriptive, unused run name:

```powershell
& .\backend\.venv\Scripts\python.exe -B .\backend\investigation\run.py --config .\backend\investigation\config.yaml --run-id validation_error_review_01
```

The default configuration is always beside `run.py`, independent of the working
directory. Input paths inside a configuration resolve against **that configuration's
directory**. The initial paths identify the completed
`paysim_three_stage_20260928_172539` selection, its final-validation report, and
`paysim_phase2_20260918` preparation. Only load trusted local joblib model bundles.

Outputs always go under this folder's `outputs/`; caches go under `runtime/`.
A generated run ID is used if omitted. An existing run ID is rejected without
overwriting it. The process prints progress to stderr, saves the same messages in
`run.log`, and prints report/manifest paths as JSON on successful completion.
Open the resulting `report.html` directly in a browser. Share the **entire run
folder**, not just the HTML file, to retain charts, case explanations, and tables.

Configuration fields:

| Field | Default | Meaning |
|---|---|---|
| `schema_version` | 1 | Supported investigation configuration format |
| `models`, `prepared`, `report` | Existing local bundles | Matching saved inputs |
| `sample_per_group` | 5 | Maximum cases per outcome/type cell |
| `seed` | 42 | Reproducible case-sampling seed |
| `minimum_denominator` | 20 | Support below which a metric is flagged sparse |

SHAP uses the selected bundle's existing policy: probability-space explanations,
the shared 200-record original-training background, and its saved seed/additivity
tolerance. Case sampling does not replace or rebalance that background.

## What the report contains

- All eight paired outcomes, including groups with zero cases.
- Subgroup confusion counts, precision, recall, F1, MCC, PR-AUC **(Average Precision)**,
  supplementary accuracy, and false alarms per 10,000 legitimate records.
- Type, amount, simulated hour/day, zero-amount and merchant-indicator analyses;
  type-by-amount and type-by-hour comparisons.
- Fixed amount bands: `<1,000`, `1,000–<10,000`, `10,000–<100,000`,
  `100,000–<1,000,000`, `≥1,000,000`. Values are dataset units, not asserted PHP amounts.
- A 1%–100% threshold sweep and separate per-model best-grid F1 diagnostics.
  Exact F1 ties prefer proximity to 50%, then the higher cutoff.
- Maximum recall within each model's current false-alarm budget, evaluated at
  distinct scores. Tied scores stay together. TP ties minimize FP, then prefer the
  higher threshold. A no-alert point is an explicit rule with a blank cutoff.
- Within-outcome score distributions and quantiles. Histogram curves are normalized
  separately by outcome, so heights do not indicate population prevalence.
- Paired SHAP explanations for deterministic samples of each nonempty outcome/type
  cell, with coverage, signed contributions, and score-reconstruction checks.

The main analysis uses the saved cutoff (currently 0.43). Diagnostic threshold
comparisons never alter saved configuration, scores, predictions, or application
behavior. Amounts are joined from verified retained source attributes; missing
amounts, if present in a compatible preparation, use the saved training median and
are explicitly flagged. Source scanning retains only validation-member rows.

### Support and interpretation

Every exported metric includes its support and a sparse flag. Precision uses TP+FP;
recall and AP use TP+FN; F1 uses 2TP+FP+FN; accuracy uses all subgroup rows; the false
alarm rate uses TN+FP. For MCC, support is the smallest of its four marginal counts,
not the square-root product in the coefficient formula. Supports are descriptive
checks, not effective sample sizes or confidence intervals. Undefined values have
blank numeric fields and explicit reasons; they are never silently replaced by zero.

Sparse metrics are retained in tables but excluded from highlighted rate findings.
Count summaries are explicitly distinguished from relative rates. Subgroup
comparisons are exploratory, with no significance or causal claims. These validation
results have already been inspected and are not unbiased final test estimates.

SHAP samples deliberately overrepresent uncommon outcomes. They are case studies,
not global feature importance. Correct cases are sampled controls, not matched
training neighbors. SHAP explains how the fitted model computed a score; it does
not prove that SMOTE caused the error. Automatic narrative is deterministic and
references supporting tables; no language-model service is called.

## Files and provenance

Each `outputs/<run-id>/` contains:

- `report.html`, `SUMMARY.md`: readable findings and interpretation.
- `tables/*.csv`: complete metric/support tables, threshold and budget results,
  distributions, outcome counts, sampled cases, coverage, and SHAP contributions.
- `validation_details.parquet`: all joined validation rows, saved scores/predictions,
  and diagnostic attributes/outcomes; model inputs are not modified.
- `charts/*.svg`, `explanations/<source-row>/<model>/`: local figures and portable
  explanation JSON/figures.
- `manifest.json`: completion state, settings, timestamps in UTC+08:00, code/package
  identities, input/output hashes, selected model identity, policy, and verification.
- `run.log`: progress. Failed runs also retain `traceback.txt` and an explicitly
  incomplete HTML page.

The existing loaders verify bundle/report fingerprints and recompute frozen selection
decisions. Additional checks enforce exact validation membership, aligned labels,
dataset/row transaction identities, saved cutoff consistency, metrics recomputation,
and sampled SHAP score parity. Input fingerprints are compared again at completion.
The inventory includes the selected fitted model files, selected-run evidence, report,
and preparation files used here; it deliberately excludes test feature files.

Interrupted runs retain a `running` manifest if termination prevents cleanup. Only
`status: complete` is usable. Run again with a new ID; compatible complete SHAP cache
entries are reused, while corrupted cache entries fail visibly. Generated outputs,
runtime caches, and logs are Git-ignored; code/configuration/tests are tracked.

## Tests

From the repository root:

```powershell
& .\backend\.venv\Scripts\python.exe -B -m pytest -c .\backend\investigation\pytest.ini .\backend\investigation\tests -q
```

Tests create unique temporary directories inside `runtime/pytest/`. They cover
eight-way outcomes, amount/time boundaries, empty outcomes, undefined/sparse metrics,
exact threshold counts/ties, score-tied budgets, deterministic sampling, identity and
provenance rejection, test-report rejection, explanation parity/additivity, incomplete
runs, non-overwriting output, and offline report references. The small integration
fixture substitutes the expensive SHAP engine; the actual investigation verifies the
shared engine against real saved predictions.

Run the existing backend regression suite separately, using an unused temporary
directory within this folder:

```powershell
$investigationTestTemp = Join-Path (Get-Location) ("backend/investigation/runtime/backend-tests-" + [guid]::NewGuid().ToString("N"))
& .\backend\.venv\Scripts\python.exe -B -m pytest .\backend\tests --basetemp $investigationTestTemp -q
```

Expected current-run acceptance counts are 252 shared fraud detections, 81
RF-SMOTE-only detections, 489 shared misses, 49 shared false alarms, and 397
RF-SMOTE-only false alarms. RF-only detections/false alarms are both zero. These are
checks for the supplied validation run, not assumptions imposed on future reports.

Independently verify a completed run's hashes, current-run counts, score-tied budgets,
and all exported SHAP reconstructions:

```powershell
& .\backend\.venv\Scripts\python.exe -B .\backend\investigation\tests\verify_saved_run.py .\backend\investigation\outputs\validation_investigation_20261115 --expect-current-counts
```

Optional browser QA uses the repository's **existing frontend Playwright dependency**
and installed Chrome. It is a development check, not a runtime requirement for this
workflow. It checks offline loading, a paired SHAP case, and mobile overflow, and saves
screenshots under `runtime/qa/`:

```powershell
node .\backend\investigation\tests\preview_report.cjs .\backend\investigation\outputs\validation_investigation_20261115
```
