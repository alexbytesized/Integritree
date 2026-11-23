# Isolated ordinary-SMOTE ratio comparison

For current execution status and the exact continuation steps, read the
[handoff and implementation plan](HANDOFF_IMPLEMENTATION_PLAN.md). The first real
run name shown below is already occupied by a preserved pre-training memory failure;
use the new run name specified in the handoff when continuing.

One command trains and evaluates the three approved RF-SMOTE ratios: **1:200,
1:300, and 1:500 fraud:legitimate**. The existing RF and 1:100 RF-SMOTE remain
unchanged references. This is a small, prespecified validation experiment, not a
guarantee that SMOTE will outperform RF.

All new code, configuration, tests, fitted artifacts, logs, reports and caches live
here. The runner imports the existing backend and the read-only diagnostic helpers
in `../investigation/`; keep both available. It does not change application APIs,
active model selection, original artifacts, or the earlier investigation.

## Run

Use the existing backend Python 3.12 environment and locked dependencies; no new
packages or network access are required. See [backend setup](../README.md).

From the repository root in PowerShell:

```powershell
& .\backend\.venv\Scripts\python.exe -B .\backend\smote_ratio_experiment\run.py --run-id ordinary_smote_ratios_20261116
```

The default `config.yaml` sits beside `run.py`, regardless of the current working
directory. `--config <path>` accepts another configuration; its input paths resolve
against that file's directory. `--run-id` is optional. Each run gets a new directory
under `outputs/`; an existing name is rejected. Training is sequential, using one
worker. Three full-size fits may take tens of minutes. Progress goes to stderr and
`outputs/<run-id>/run.log`.

The first version accepts exactly the approved three ratios, in order, and seed 42.
It offers no forest-parameter or seed search. Before any fit, it verifies that each
ratio adds minority records to the original training partition and performs the
backend's conservative memory preflight. That estimate covers working arrays, not
a guarantee of peak tree/runtime memory. Failure stops the experiment explicitly.

## Fixed training protocol

- Same original training/validation partitions and saved 11-feature preprocessing.
- Ordinary SMOTE, five neighbors, seed 42. The ratios describe the **final**
  minority-to-majority class proportions; integer targets follow imbalanced-learn.
- Original rows and labels are preserved exactly. Synthetic indicator fractions
  are preserved and audited, following the existing ordinary-SMOTE implementation.
- Forest: 100 trees, maximum depth 10, minimum leaf size 1, model seed 42, one
  worker. Every remaining setting matches the verified RF benchmark.
- Only three new RF-SMOTE forests are fitted; RF and the existing 1:100 model are
  loaded and their saved validation scores replayed. No preprocessing refit,
  validation resampling, test-set evaluation, deployment or automatic promotion.

This adjusts sampling intensity and decision cutoffs; it does **not** fit a
probability calibration model such as isotonic regression or Platt scaling.

## Shared cutoff

For **each** candidate, all 100 cutoffs from 1% to 100% are applied equally to RF
and that candidate. Select the largest **mean of their validation F1 scores**.
The shared backend routine uses exact rational comparisons; exact ties prefer the
cutoff closest to 50%, then the higher cutoff. Scores equal to the cutoff predict
fraud without rounding.

The selected cutoff is applied equally to RF, the candidate and the original
1:100 reference in that comparison. The original 1:100 reference does not influence
the selection. Different ratio comparisons may select different common cutoffs;
there is never a different RF and candidate cutoff **within** a comparison.

All five models are also reported at the original **43% cutoff**. Use that table
to compare sampling effects at a constant decision rule. The selected-cutoff tables
include the additional effect of cutoff selection. No ratio is automatically
chosen for deployment.

## Evidence and interpretation

Each run contains:

- `report.html`, `SUMMARY.md`: combined comparison, linked to all candidate reports.
- `overall_43.csv/.parquet`, `overall_selected.csv/.parquet`, `changes_at_43.csv`:
  precision, recall, F1, MCC, Average Precision (`pr_auc`), accuracy, confusion counts,
  false alarms per 10,000 legitimate records, undefined reasons and sparse flags.
- `ratio_<denominator>/model/`: `rf_smote.joblib`, copied preprocessing,
  configuration, sampling audit, synthetic features, reload verification and hashes.
  These use kind `isolated_smote_ratio` and the **local loader** in
  `ratio_experiment/core.py`. They cannot be loaded as ordinary application bundles.
- `ratio_<denominator>/predictions_43.csv/.parquet` and
  `predictions_selected.csv/.parquet`: identical validation identities, true labels,
  scores and decisions for the RF/candidate/reference comparison. Parquet also
  contains diagnostic attributes and paired outcomes.
- Candidate `tables/`: all eight paired outcomes, transaction type, amount,
  simulation hour/day, zero amount, merchant flags, type-by-amount and type-by-hour
  subgroup metrics at both cutoffs; threshold grid and score distributions.
- Candidate `explanations/`: up to five sampled cases per candidate-versus-RF
  outcome/type group, stable row ordering and seed 42. Both models explain the same
  cases using a shared 200-record original-training SHAP background. Score parity
  and additive reconstruction are checked. Each ratio may have a different sample
  because its error groups differ. These are **sampled case studies**, not global
  importance or evidence that synthetic records caused specific errors.
- `manifest.json`, `run.log`: status, protocol, fits started, completed ratios,
  input/code/dependency hashes, source preservation and output fingerprints.

Open `report.html` directly in a browser. Share the entire run folder to preserve
all local links. No CDN, server, internet or remote script is needed. Runtime SHAP
and plotting caches stay under `runtime/`.

Amount bands are `<1,000`, `1,000–<10,000`, `10,000–<100,000`,
`100,000–<1,000,000`, and `≥1,000,000`, in dataset units. Hours/days are simulation
indexes, not actual clock dates. Undefined metrics remain blank with reasons.
Supports below 20 are flagged; subgroup comparisons are exploratory and should
not be treated as significance tests. Counts and rates answer different questions.

The validation set has already informed previous work. Three ratios and many
cutoffs are examined here, so the highest observed F1 is development evidence,
not an unbiased generalization estimate. Both improvements and regressions remain
in the report. F1 remains the primary discussion metric.

## Failures and reproducibility

Only `status: complete` means all three fits, evaluations, sampled explanations,
offline link checks and source-preservation checks passed. Failures retain an
incomplete report, log and traceback. Abrupt termination may leave `running` status;
treat it as incomplete. There is no automatic retry or resume that could silently
add fits. Review a failure before authorizing a new run.

The input snapshot includes training/validation features, source attributes,
membership, selected reference evidence and the validation report. It excludes
test feature files. Source artifacts are hashed again at completion. Background
sampling reads original training only. Model deserialization is for trusted local
artifacts only; hashes detect accidental changes, not malicious pickle files.

## Verification

Experiment tests, with temporary files confined to `runtime/`:

```powershell
& .\backend\.venv\Scripts\python.exe -B -m pytest -c .\backend\smote_ratio_experiment\pytest.ini .\backend\smote_ratio_experiment\tests -q
```

The integration test runs all three ratios on a tiny synthetic fixture, using real
SMOTE, random forests, SHAP and report generation. It does not use real PaySim data.
Unit tests cover exact shared-cutoff selection and ties, all eight outcome groups,
amount boundaries, sparse/undefined metrics, identities, deterministic sampling,
provenance rejection, non-overwriting runs and failed status.

Relevant shared backend regression checks:

```powershell
$ratioTestTemp = Join-Path (Get-Location) ("backend/smote_ratio_experiment/runtime/backend-tests-" + [guid]::NewGuid().ToString("N"))
& .\backend\.venv\Scripts\python.exe -B -m pytest .\backend\tests\test_training.py .\backend\tests\test_preprocessing.py .\backend\tests\test_evaluation.py .\backend\tests\test_staged_selection.py .\backend\tests\test_shap_adapter.py --basetemp $ratioTestTemp -p no:cacheprovider -q
```

Independent acceptance of a completed run, including full validation score replay:

```powershell
& .\backend\.venv\Scripts\python.exe -B .\backend\smote_ratio_experiment\tests\verify_saved_run.py .\backend\smote_ratio_experiment\outputs\ordinary_smote_ratios_20261116
```

Optional browser QA uses the already installed frontend Playwright and Chrome,
without downloading anything. Screenshots and results stay in `runtime/qa/`:

```powershell
node .\backend\smote_ratio_experiment\tests\preview_report.cjs .\backend\smote_ratio_experiment\outputs\ordinary_smote_ratios_20261116
```
