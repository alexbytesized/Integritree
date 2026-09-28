# Evaluation, three-stage selection, and explanations

The sole selection workflow is [three-stage validation](THREE_STAGE_VALIDATION.md):
SMOTE ratio by RF-SMOTE validation Average Precision, shared forest settings by
mean validation AP, then a common threshold by exact mean validation F1. Each
stage freezes before the next. Both models are retained; selection does not pick
a model winner. Stage 1 froze 1:100; Stage 2 selected 100 trees/depth 10/leaf 1.
Stage 3 uses exactly 1%, 2%, ..., 100% as threshold candidates. See
[run status](VALIDATION_RUN_STATUS.md) for measured outcomes.

## Approved and implemented methods

- Fraud is class 1. Both models use the same records and common threshold;
  exact score/threshold ties predict fraud.
- Precision, recall, direct-count F1, MCC, Average Precision (reported as
  **PR-AUC (Average Precision)**), and supplementary accuracy are exported.
- Undefined precision/recall/F1/MCC use JSON null plus a reason. F1 is zero when
  its count-based denominator is positive but TP is zero, even if precision is
  undefined. AP is unavailable with no actual positives; with all positives AP
  is 1, although such a population is not a representative fraud benchmark.
- Signed symmetric differences use 100*(S-B)/((S+B)/2). Both zero means unavailable;
  if either MCC is negative, use S-B in coefficient units. Legacy absolute
  policies remain identifiable. These comparisons do not test significance.
- Primary McNemar uses (abs(b-c)-1)^2/(b+c), chi-square df=1, alpha=.05.
  The paired table labels b=RF-only correct and c=RF-SMOTE-only correct.
  For 1–24 discordant pairs, also report the exact two-sided binomial p-value
  as supplementary, without replacing the primary test. With zero discordances,
  use p=1 by convention, null statistic, and “No discordant pairs.”
- McNemar concerns paired classification error rates; it does not establish
  significance for each reported metric. Nonsignificance is not equivalence.

## Selection and test isolation

`staged_selection.py` implements the [canonical protocol](THREE_STAGE_VALIDATION.md).
Stage 1 selects the ratio by RF-SMOTE AP, Stage 2 selects a shared forest by mean
AP, and Stage 3 freezes the common threshold by exact mean F1. A schema-2 selected
bundle references immutable candidate models, reports, decisions, and hashes.
The loader independently checks the frozen evidence and selection rules.

`--stop-after-stage 1` stops before forest selection; `--stop-after-stage 2`
stops after the frozen forest decision and before threshold search. A graceful
pause request finishes the active paired candidate and its validation checkpoint,
then records `paused`. Stage 2 reviews recompute candidate metrics from verified
reports and include a snapshot of the run metadata for later audit. A partial run
cannot be loaded as a final model pair. Official test evaluation requires a
completed selection. Ordinary fitted pairs remain readable for validation and
inference; obsolete two-step selections are unsupported. There is no refit on
combined partitions. Interrupted runs verify the frozen plan and recover completed
fits and reports; one process owns the run lock at a time.

## SHAP reference, computation, and numerical compatibility

Both models use interventional TreeSHAP in fraud-probability space. Their shared
reference is a uniform sample without replacement of 200 **original training**
records, seed 42, without class balancing. Store the selected identities and
hashes. Small synthetic fixtures use all rows when fewer than 200 exist. An
explicit masker prevents SHAP's default reduction to 100 background records.

On-demand explanations include all eleven features, original readable derived
values, numerical attributions, deterministic prose, the largest contribution
above 1e-9, and a waterfall SVG. Negative influences are included. A legitimate
prediction may still have positive contributors; all-zero/negative contributions
produce an explicit no-positive state. No ground truth or excluded balance field
is used to explain a prediction. Contributions are associations with the model
output, not proven causes of fraud.

Every explanation must satisfy abs(base + sum(contributions) - fraud_score) <=
1e-6. A failure stops the explanation; contributions are never rescaled to force
agreement. The saved model remains the source of the prediction and risk score.

Compatibility work was necessary for the pinned SHAP 0.52.0 interventional
kernel: it stores float32 thresholds and signed 16-bit node indexes. Unadjusted
threshold rounding failed the synthetic reconstruction check, and large forests can exceed its node-index range.

`shap_adapter.py` supplies an equivalent internal representation to TreeExplainer:

1. Cast reference/explanation inputs to float32, as scikit-learn RF inference does.
2. Round each threshold **down** to the float32 grid. For every representable
   float32 x, x <= original_threshold iff x <= floored_threshold. This preserves
   the branch predicate instead of using nearest rounding, which can change it.
3. Express a large tree as a sum of components of at most 8,192 nodes. Each keeps
   its original subtree plus the ancestor conditions, returning zero outside its
   assigned path. Original leaf probabilities retain their 1/number-of-trees
   weighting. The component sum is exactly the original forest prediction.
4. Use TreeExplainer with the same interventional reference and probability target.
   Shapley linearity preserves the attributions of this additive representation.

This does not retrain, prune, approximate, or edit the saved classifiers. It
supports finite inputs and the approved depth-10/depth-20 forests. Tests compare
every attribution with an independent exhaustive coalition calculation on a
small synthetic forest, force tree decomposition, check prediction equivalence,
and verify that original thresholds are unchanged. Keep these checks when
upgrading SHAP; do not simply remove the adapter or additivity checks.

Reference: [SHAP interventional kernel source](https://github.com/shap/shap/blob/v0.52.0/shap/cext/tree_shap.h).

Cache keys include the analysis, record ID, exact features, model metadata,
reference metadata, SHAP version, policy, and implementation/adapter hashes.
Existing complete entries are fingerprint checked. This local Python interface
is not yet a concurrent HTTP job service.

## Coverage and outputs

- Evaluation covers every record in the selected split and writes paired CSV and
  Parquet predictions, metrics, comparisons, statistical tests, configurations,
  confusion-matrix/precision-recall SVGs, and a provenance manifest. Parquet is the
  canonical numerical export; when reading CSV with pandas for exact score reuse,
  specify `float_precision="round_trip"` to preserve the written 17-digit values.
- Default global SHAP coverage is a uniform 1,000-record sample without replacement
  from the evaluation report, seed 42, shared across models. Record the sample IDs,
  actual class counts, and population size. Do not describe a sample's mean absolute
  contributions as explanations for all records or as a fraud-only summary.
- `--scope preview --limit N` is a smaller development check. `--scope full` is an
  explicit full-report job; it is never triggered by computing metrics.
- Explanation reports contain JSONL, selected-record CSV, per-model mean absolute
  contributions, and a coverage/provenance manifest. Local cached waterfall assets
  are referenced from those explanations. Portable application downloads are Phase 5.

## Verification and execution

Synthetic checks cover metric edge cases, exact selection, freezing, interruption
recovery, candidate memory release, model reuse and corruption rejection, held-out
guards, inference parity, and SHAP reconstruction/coalition equivalence.
See [current run status](VALIDATION_RUN_STATUS.md) for the latest test count and
real-data Stage 1 evidence. Software checks do not establish fraud-detection
performance. Current execution includes no official test or SHAP job.

See [backend commands](../backend/README.md) and
[manuscript changes](THESIS_DOCUMENT_CHANGES.md).
