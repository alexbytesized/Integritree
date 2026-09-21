# Integritree help modals and limitations-page copy

Updated: 2026-09-22

Status: approved help/limitations content, not a completed frontend implementation.
Signed symmetric percentage difference and its edge-case display are approved.
Conditional alternatives do not silently settle an open methodology choice. Link this document from
[the decision record](MOCKUP_EVALUATION_DECISIONS.md).

## Shared modal pattern

Use the same order in every modal:

1. Title (full name where appropriate).
2. What it means (one or two short sentences).
3. Formula or How to read it (whichever is meaningful).
4. Interpretation.
5. Keep in mind (one material limitation).

Use a labelled dialog with keyboard-operable close, Escape support, managed focus,
and focus returned to the triggering button. Use button semantics on clickable
labels, not hover-only text. Formulas should be selectable/readable text or
accessible math, not only images. Do not rely on color for statistical decisions.

Backend must supply method identifiers so conditional formula copy matches the
actual calculation. Until that calculation exists, show Not calculated rather
than substituting random values. Internal documentation alternatives below are
not all displayed simultaneously to ordinary users.

### Shared notation

Fraud is the positive class. For the selected model and evaluation set:

- TP: actual fraud predicted as fraud.
- FP: actual legitimate predicted as fraud.
- TN: actual legitimate predicted as legitimate.
- FN: actual fraud predicted as legitimate.
- S: RF-SMOTE metric value; B: benchmark RF metric value.

Show relevant symbol definitions in each modal or provide an expandable shared
legend. Ground truth is independent of model predictions. Calculations use the
declared evaluation set, not merely the visible table page or a filtered subset.

## Precision

**What it means**

Of the transactions this model predicted as fraud, how many were actually labeled fraud?

**Formula**

Precision = TP / (TP + FP)

**Interpretation**

A higher value means a greater proportion of fraud predictions were correct.
The value ranges from 0 to 1, or 0% to 100% when displayed as a percentage.

**Keep in mind**

Precision does not show how many actual fraud cases the model missed. Read it
alongside recall. If there are no fraud predictions, the denominator is zero;
the result must follow the documented undefined-value policy.

Source: [scikit-learn precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html).

## Recall

**What it means**

Of the transactions actually labeled fraud, how many did this model identify as fraud?

**Formula**

Recall = TP / (TP + FN)

**Interpretation**

A higher value means a greater share of actual fraud cases was detected.
The value ranges from 0 to 1, or 0% to 100%.

**Keep in mind**

Recall does not show how many legitimate transactions were falsely flagged.
Read it alongside precision. If no actual fraud cases are present, the denominator
is zero and the documented undefined-value policy applies.

Source: [scikit-learn recall](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html).

## F1-Score

**What it means**

F1 combines precision and recall into their harmonic mean, giving a single measure
of the balance between reliable fraud predictions and fraud detection coverage.

**Formula**

F1 = 2 * TP / (2 * TP + FP + FN)

Equivalent when defined: F1 = 2 * Precision * Recall / (Precision + Recall).

**Interpretation**

A higher value indicates a stronger balance between precision and recall.
The value ranges from 0 to 1, or 0% to 100%.

**Keep in mind**

F1 does not use true negatives directly. Consult MCC and the confusion matrix as
well. If 2*TP+FP+FN is zero, apply the documented undefined-value policy.

Source: [scikit-learn F1](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html).

## MCC - Matthews Correlation Coefficient

**What it means**

MCC measures association between predicted and actual labels using all four
confusion-matrix outcomes. It is useful when fraud and legitimate counts differ greatly.

**Formula**

MCC = (TP * TN - FP * FN) /
      sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

**Interpretation**

- +1: perfect agreement with the actual labels.
- 0: no correlation between predicted and actual labels, where the coefficient is defined.
- -1: complete inverse agreement in a nondegenerate binary evaluation.

Display a coefficient, not an unexplained percentage. Higher is better.

**Keep in mind**

MCC is not accuracy. Zero does not mean zero correct predictions or prove that a
model is randomly guessing. A zero denominator is a degenerate case; clearly
document any implementation convention instead of treating a fallback zero as
an ordinarily defined correlation.

Source: [scikit-learn MCC](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html).

## PR-AUC - Area Under the Precision-Recall Curve

**What it means**

PR-AUC summarizes the trade-off between precision and recall as the fraud-score
threshold changes, instead of measuring only one classification threshold.

**Formula**

Conceptually: PR-AUC = integral from recall 0 to 1 of Precision(recall) d(recall).

The numerical formula shown must match the method used. Include Method: {method_label}.

If trapezoidal integration is selected, display:

PR-AUC = sum_i (R_i - R_(i-1)) * (P_i + P_(i-1)) / 2

Here the curve points are ordered in increasing recall, and P_i/R_i denote precision
and recall at successive thresholds. Implementation must handle curve orientation
correctly and preserve its points.

If average precision is selected, label the metric PR-AUC (Average Precision) or
Average Precision (AP), and display:

AP = sum_i (R_i - R_(i-1)) * P_i

**Interpretation**

Higher values indicate stronger precision-recall performance on the same evaluation
population. Values range from 0 to 1; a percentage display is optional.

**Keep in mind**

This metric requires fraud scores and actual labels. It cannot be derived from one
confusion matrix alone. AP and trapezoidal integration are different conventions;
do not display one formula for the other. Class prevalence affects interpretation
and comparisons across datasets.

Developer status: the configured PR-AUC method remains unresolved. This draft does
not choose one. Suppress computed-result claims until the method and class/undefined
policies are finalized; do not publish two alternatives as if both generated one value.

Source: [scikit-learn AP conventions](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html).

## Signed Percentage Difference

Approved on 2026-09-22. The title, formula, and sign guide must agree. The active
experiment setting is signed_over_mean, with legacy absolute_over_mean loading
preserved. The calculation and manuscript alignment remain pending. Do not label
legacy absolute results signed.

**Title:** Signed Percentage Difference

**What it means**

Compares RF-SMOTE with benchmark RF using the average of their metric values as
the reference, while keeping the direction of the difference.

**Formula**

Signed difference (%) = 100 * (S - B) / ((S + B) / 2)

Use this for nonnegative metric values with a positive denominator.

**Interpretation**

- Positive: RF-SMOTE has the higher metric value.
- Negative: benchmark RF has the higher metric value.
- Zero: equal metric values.

**Keep in mind**

This is descriptive, not a significance test. It is not the same as percentage
change relative to benchmark RF, which uses B alone in the denominator. If the
mean is zero, the percentage is undefined. For negative MCC comparisons, show
S-B in coefficient units instead; do not silently change the percentage formula.

### Distinguishing other quantities (author guidance)

- Relative change from benchmark (%) = 100*(S-B)/B, meaningful only under an
  appropriate nonzero baseline policy. This is a different denominator.
- Percentage-point change = 100*(S-B) when S and B are stored as 0..1 proportions.
- MCC coefficient difference = S-B, not a percentage-point change in accuracy.

For S=0.90 and B=0.80: signed symmetric difference is +11.76%, relative change is
+12.5%, and the percentage-point change is +10 points. These are explanatory
examples, unrelated to screenshot placeholders.

## McNemar's Test

**What it means**

Tests whether the two models have different classification error rates when they
are evaluated on the same labeled transactions.

**Formula**

Let b be the count where benchmark RF is correct and RF-SMOTE is incorrect.
Let c be the count where benchmark RF is incorrect and RF-SMOTE is correct.

For the manuscript's continuity-corrected chi-square version, when b+c>0:

chi-square = (abs(b - c) - 1)^2 / (b + c)

Compare this statistic to the chi-square distribution with one degree of freedom
to obtain the approximate p-value. Show Method: continuity-corrected chi-square.

If an exact method is selected instead, replace that formula with:

n = b+c; X ~ Binomial(n, 0.5)
p = min(1, 2 * Pr(X <= min(b,c)))

Show Method: exact binomial. Do not label its returned statistic as chi-square.
Small-discordance handling remains open. The approved zero-discordance display
convention is p=1, no statistic, with No discordant pairs; do not divide by zero.

**Interpretation**

Compare the p-value with the declared significance level, alpha=0.05:

- p < alpha: reject the null hypothesis of equal error rates.
- p >= alpha: fail to reject the null hypothesis.

**Keep in mind**

The test does not separately establish significance for precision, recall, F1,
MCC, or PR-AUC. A significant result can favor either model. The direction of the
error difference comes from b versus c, not the p-value alone. Evidence is scoped
to the paired evaluation population and its sampling assumptions.

Source: [statsmodels McNemar](https://www.statsmodels.org/stable/generated/statsmodels.stats.contingency_tables.mcnemar.html).

## Contingency Table

**What it means**

Groups each evaluated transaction according to whether each model's prediction
matches its actual label. Each transaction belongs to exactly one cell.

**How to read it**

| | RF-SMOTE correct | RF-SMOTE incorrect |
| --- | --- | --- |
| Benchmark RF correct | a: both correct | b: only benchmark RF correct |
| Benchmark RF incorrect | c: only RF-SMOTE correct | d: both incorrect |

Total evaluated records: N = a+b+c+d.

**Interpretation**

McNemar's test focuses on b and c. If c>b, RF-SMOTE made fewer errors on this set.
If b>c, benchmark RF made fewer errors. If b=c, their error counts are equal.

**Keep in mind**

This compares two models' correctness. It is different from a confusion matrix,
which compares one model's predicted classes with actual classes. Ground truth
and correctly paired transaction identities are required.

## P-Value

**What it means**

Assuming equal classification error rates and the test's assumptions, the p-value
measures how unusual a disagreement imbalance at least as extreme as the observed
one would be.

**Formula / guide**

For the chi-square version:

p = 1 - F_chi-square(df=1)(observed chi-square)

For the exact version, use the binomial tail formula shown in the McNemar modal.
Display only the formula corresponding to the selected method.

**Interpretation**

- p < 0.05: statistically significant evidence against equal error rates.
- p >= 0.05: insufficient evidence to reject equal error rates.

**Keep in mind**

The p-value is not the probability that the null hypothesis is true, the probability
that one model is better, or the size of an improvement. A nonsignificant result
does not prove the models are equivalent. Use scientific notation for very small
values; do not display a rounded zero as an exact zero probability.

Sources: [ASA p-value principles](https://www.amstat.org/asa/files/pdfs/p-valuestatement.pdf),
[statsmodels implementation](https://www.statsmodels.org/stable/generated/statsmodels.stats.contingency_tables.mcnemar.html).

## Undefined values and publication gates

Approved presentation policy: N/A with a specific reason when a metric is
mathematically undefined, instead of silently presenting a fallback zero as an
ordinary measurement. If the implementation uses library conventions, document
them in exports and conditional help. Never use an undefined input in a comparison.

Before publishing these modals:

- Implement the approved signed calculation and align the thesis; configuration is set.
- Select PR-AUC numerical method.
- Select McNemar variant/edge cases.
- Implement the approved undefined-value and negative-MCC comparison policies.
- Make rendered formula and exported method identifiers match backend computation.
- Use concise modal copy; developer alternatives are documentation, not a giant
  combined user-facing modal.

## Research Scope and Limitations page

Status: approved for implementation. Fill in the actual mapping conventions and
implemented retention details before publishing; those facts remain unresolved.

**Title: Research Scope and Limitations**

**About Integritree**

Integritree is a research prototype that compares Random Forest models trained
with and without SMOTE using synthetic PaySim transaction data. It displays model
predictions, scores, explanations, and evaluation results.

**What a prediction means**

A prediction describes how a trained model classifies the supplied inputs. It does
not verify that a real transaction is fraudulent or safe. The two models may disagree.

**Receipt demonstration**

Receipt images provide transaction details for an experimental demonstration.
Evaluation on PaySim does not establish prediction performance on real GCash
transactions. Only receipt workflows identified as supported by the application
can be analyzed. The system does not verify receipt authenticity or image editing.

**Timing assumptions**

PaySim uses simulation steps rather than verified receipt calendar timestamps.
The demonstration applies a documented date/time convention to construct timing
features. This mapping is an assumption, not a verified alignment with PaySim's
calendar or local clock. Display the actual configured convention here.

**Amount assumptions**

Receipt amounts are confirmed in Philippine pesos. Their numerical mapping into
the model has not been validated as equivalent to PaySim's amount units or
distribution. Applying the saved transformations does not establish that equivalence.

**Transaction and account types**

Users confirm transaction and account categories. These confirmations describe
the supplied input; they are not independent verification of a recipient's identity
or account status. Missing essential information or unsupported workflows may
prevent analysis.

**Risk communication bands**

Minimal Risk, Low Risk, Moderate Risk, High Risk, and Critical Risk are display
categories for the model score. They are not validated real-world probabilities
or operational intervention rules. Predicted class follows the model's saved
decision threshold, which is separate from these categories.

**SHAP explanations**

SHAP describes how input features influence a model's output relative to its
baseline. It does not prove the cause of actual fraud. The strongest increasing
feature can be shown even when the final prediction is legitimate.

**Research evaluation**

Evaluation requires independent labels and a clearly identified set of records.
McNemar's test compares paired classification errors; it does not test each metric
separately. Download results to see evaluation methods, model provenance, and scope.

**Personal information and retention**

Sender and recipient names are not model predictors. Masked or unavailable names
do not need to be reconstructed. Before release, this section must state the
implemented storage location, access rules, retention period, and deletion behavior.
Do not publish deletion/security promises that have not been implemented.

### Suggested entry-point copy

Receipt upload/confirmation:

> Experimental receipt demonstration. Timing and amount mappings use assumptions
> that have not been validated against real GCash outcomes. Review Research Scope
> and Limitations before interpreting the results.

Receipt results:

> These are model predictions, not verified findings of fraud. See Research Scope
> and Limitations for the assumptions behind this demonstration.

Link the page from the footer, relevant receipt screens, and the exported report.
Keep a short contextual notice at the point of use. A separate page is useful
documentation; it does not itself make an incompatible mapping scientifically valid.
