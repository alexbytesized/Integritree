# Integritree mock-up evaluation and implementation decisions

Updated: 2026-09-22

## Purpose and status

This is the separate decision record requested by the researchers. It tracks
the mock-up review, frontend requirements, backend dependencies, proposed
resolutions, and acceptance criteria. It is not a manuscript revision or evidence
that a feature has been implemented.

Use the corrected researcher-results mock-up, which includes McNemar's test.
Ignore every numerical value in the screenshots: all are illustrative placeholders.
Numerical ranges explicitly approved below are design decisions, not screenshot findings.

Status definitions:

- **Agreed:** explicitly accepted or specified by the researchers in this exchange.
- **Existing requirement:** already established by the thesis/project records.
- **Proposed:** recommended here; not yet accepted as a design or methodology decision.
- **Open:** requires a specific choice before dependent implementation.

Implementation status: the review is recorded for mock-up revision and backend
planning. The 2026-09-22 audit also aligned the active comparison-method configuration
with signed_over_mean while preserving legacy configuration loading. Evaluation,
SHAP, application APIs, and receipts remain future work. Trained artifacts and the
manuscript are unchanged. See [the backend alignment audit](BACKEND_ALIGNMENT_AUDIT.md).

## Decision ledger (matches review numbering)

| ID | Status | Decision or next action |
| --- | --- | --- |
| 1 | Existing requirement | Retain both models, five metrics, confusion matrices, descriptive comparison, and the already-present McNemar section. |
| 2 | Agreed | Add a downloadable CSV template. Schema help and validation behavior below implement the previously reviewed input requirements. |
| 3 | Agreed scope / input details proposed | Keep GCash person-to-person transfers. Show five transaction-type options with only TRANSFER selectable; other four visibly unsupported/disabled. Timing and amount limitations acknowledged; retain available names/reference. Entity dropdown and optional-field recommendations below remain under review. |
| 4 | Agreed | Editable extracted fields, required indicators, extraction/correction status, validation errors, replace-image action, unsupported states, and explicit confirmation. Manual-entry fallback remains open. |
| 5 | Agreed | Top positive contributor, deterministic explanation, and View SHAP details with a waterfall only. Numerical table is a suggestion, outside the implementation plan. Technical background size/seed and runtime policy still need finalization. |
| 6 | Agreed | Apply table best practices but retain the existing color-only visual model association for scores/contributors. Visible model badges/subheaders remain suggestions only. No risk sorting; dataset record ID and horizontal scroll retained. Use Model and Prediction outcome filters as specified below. |
| 7 | Agreed | Use Original Inputs and Derived Inputs sections, show researcher ground truth, preserve return state. Do not add separate correctness/outcome badges or on-screen model-run details. |
| 8 | Agreed | Predicted Fraud / Predicted Legitimate; Minimal, Low, Moderate, High, Critical communication bands at 20-point boundaries. |
| 9 | Agreed | Download results as the ZIP package specified below. Put evaluation context/model provenance in the exported package, not a new on-screen evaluation panel. |
| 10 | Agreed presentation / remaining technical choices | Use signed symmetric percentage difference, standardized help modals, and the comparison/undefined-value policies below. PR-AUC integration and small-discordance McNemar handling still need a specific method choice. |
| 11 | Requested draft | Comprehensive landing-page copy is provided below for review. |
| 12 | Agreed direction | Implement exports, processing/error states, result actions, accessibility, responsive layouts, receipt handling, and usability checks. Exact formats, retention duration, and job lifecycle remain open where specified. |

The earlier database discussion established that an application database is not
technically mandatory. It did not authorize a particular persistence design.

## 2. Researcher CSV upload

Frontend:

- Add Download CSV template and Required columns actions.
- The template describes the raw PaySim schema. Identify the five predictor
  sources (step, type, amount, nameOrig, nameDest), the separate isFraud label,
  and the excluded balance/flag fields. Keep template and ingestion schema aligned.
- Explain that arbitrary labeled e-wallet CSVs are not automatically compatible.
- Show file preview, validation progress, accepted/rejected records, and actionable
  row/column errors. Do not expose unrelated record contents in error messages.
- For evaluation, require independent labels. Do not infer or impute ground truth.
- A sample/template file is demonstration input, not an official held-out test set.

Backend:

- Reuse source validation and training-fitted preprocessing.
- Preserve source record identity and label alignment across both models.
- Excluded raw columns may remain in source records but never enter predictors.
- Do not train, refit preprocessing, or apply SMOTE to uploaded prediction records.
- Unlabeled batch inference is optional scope; if added, evaluation is unavailable.

## 3. Experimental receipt mapping and input review

Keep official evaluation on held-out PaySim records. Keep the receipt path separate
and identify it as an experimental demonstration with no established real GCash
performance. This mapping proposal does not validate transfer between domains.

### Proposed user inputs and mapping

| Receipt input | Proposed treatment | Qualification |
| --- | --- | --- |
| Provider/layout | Accept supported GCash person-to-person transfer layouts only. | Exact layout list/OCR engine remain open. |
| Date and time | Confirm Asia/Manila local date/time; derive hour 0..23 and weekday Monday=0 through Sunday=6. | A demonstration convention: PaySim cycle day 0 is not known to be Monday and its clock has no verified Philippine anchor. |
| Transfer amount | Parse the confirmed transfer principal, separately from fees; retain currency PHP. Derive log1p(amount) and the zero-amount indicator. | Passing the PHP numeric amount unchanged is an explicit demonstration assumption, NOT a verified conversion into PaySim units. Do not invent exchange rates or infer currency equivalence. |
| Transaction type | Map confirmed person-to-person transfer to TRANSFER. | Reject merchant payments, cash-out, and other unsupported workflows. |
| Sender/recipient account type | Confirm both are personal accounts, then set both merchant flags to 0. | Do not classify account types from names alone. Unknown or contradictory account type blocks the supported workflow. |
| Names/reference | Retain only as needed for confirmation and traceability. | Never use personal names, phone numbers, or receipt reference as predictors. |

The researchers acknowledged the timing and currency assumptions on 2026-09-22.
This does not establish empirical validity. The precise calendar convention and
numeric amount-mapping details remain to be finalized. Receipt scope is confirmed
as person-to-person transfers only. If a
valid essential input cannot be established, show Unable to analyze. Do not
substitute arbitrary step values, fabricated PaySim IDs, or model explanations.

The mapping limitation is substantive: recreating step from day/hour does not
establish equivalence, and independently normalizing receipt values does not fix
the currency/distribution issue. Validated deployment would need representative
compatible labeled data; that is outside the current PaySim experiment.

### Proposed confirmation fields (2026-09-22)

| Field | Type | Validation and role |
| --- | --- | --- |
| Transaction ID / receipt reference | String | Retain leading zeros and visible text; distinguish from a generated internal analysis ID. If unreadable, recommend allowing analysis with an internal ID and marking the receipt reference unavailable. |
| Amount | Finite nonnegative number | Confirm transfer principal, not balance or fee-inclusive total; retain PHP currency metadata. Use decimal-safe parsing before model conversion. |
| Transaction Type | Enum dropdown | TRANSFER, PAYMENT, CASH_IN, CASH_OUT, DEBIT are model categories, not proof that every GCash workflow is supported. |
| Transaction Date | Date | Required for the selected timing convention. |
| Transaction Time | Time | Unambiguous AM/PM or 24-hour input; retain Asia/Manila timezone metadata. |
| Sender Type | Enum dropdown | Customer/non-merchant (C), Merchant (M), with Unknown/not shown as an unresolved UI state, not a third model class. |
| Recipient Type | Enum dropdown | Same choices; require a resolved compatible type before predicting. |
| Sender name as shown | Optional string | Preserve masking/redaction. Do not require or reconstruct full identity. |
| Recipient name as shown | Optional string | Same policy; traceability only. |

PaySim C denotes customer/client and M denotes merchant. These are entity codes,
not first letters of real names. Merchant flags are derived from confirmed types.
A cash-out agent is not automatically a merchant; unsupported roles need their own
mapping. All original training senders are C in the supplied file, so selecting a
merchant sender would be outside the observed training support for that feature.

Agreed in the 2026-09-22 follow-up: list all five transaction categories, with only
TRANSFER enabled for the initial receipt workflow. PAYMENT, CASH_IN, CASH_OUT,
and DEBIT remain visible but disabled and labelled Unsupported for now. Show a
nearby explanation because disabled options cannot reliably expose hover help.
The backend must reject these unsupported receipt types as well. Do not silently
relabel a detected payment/cash-out receipt as a transfer to fit the form.
Supporting all categories later requires layouts, type/entity-role mappings, and
validation cases for each. Research CSV processing still supports all five types.

Available masked names/reference may be retained for traceability as agreed.
Exact names are optional and never model inputs. Names must not determine C/M.

### Proposed backend adapter

- Accept user-facing receipt fields in a separate contract.
- Derive and validate the same 11 unscaled features used by the models.
- Reuse the existing saved scaling stage exactly once, with the existing feature
  order. Do not refit, clip, double-scale, or retrain as part of this adapter.
- Refactor shared preprocessing so raw PaySim and receipt paths converge before
  scaling. Keep raw PaySim behavior unchanged.
- Preserve receipt field provenance, adapter version, confirmed values, and model
  run identity with the result. Mark these predictions as experimental.
- Missing source columns are acceptable only when the complete derived feature
  vector can be validly constructed under the agreed mapping.

## 4. Confirmation form

Agreed controls: visible editing, required fields, structured transaction-type
selection, Extracted / Corrected / Missing states, field-level errors, Replace
image, and Confirm details and analyze. Disable analysis for unresolved essential
inputs. Unsupported/unreadable images get an explanation and recovery action.

The user should not have to enter step, one-hot indicators, or scaled values.
Date supplies the weekday; no duplicate weekday entry is necessary when date is
available. Clarify AM/PM or use an unambiguous time input.

## 5. Agreed SHAP presentation; technical details to finalize

### Three presentation levels

1. Top risk-increasing contributor on the transaction table/result card.
2. A short, deterministic explanation with increasing and decreasing influences.
3. View SHAP details: model-specific waterfall graph only.

SHAP supplies numerical attributions; its plotting functions can visualize them.
It does not automatically write a reliable natural-language explanation. A graph
is a visualization of the raw values, not itself the raw result.

Backend responsibility: compute SHAP values and the narrative, call the SHAP
waterfall plot function, and return a chart asset and its explanatory metadata.
Keep the underlying numerical values for validation, narrative generation, and
reproducibility; this does not require a frontend numerical table.
Frontend responsibility: request the selected record/model, open an accessible
modal, display the supplied graphic and accessible description, and handle
loading/errors. Keep the numerical table outside the implementation plan.

Suggestion only, deferred on 2026-09-22: a numerical SHAP table. If later selected,
the frontend controls its columns, labels, spacing, colors, and display
precision. It uses the backend's feature values, signed SHAP contributions,
baseline, and explained output without recalculating them. Preserve full supplied
precision in raw exports; display rounding must not change contributor selection
or the narrative. Explain the units of contributions and retain their signs.

### Definition of the top contributor (agreed)

Explain fraud-class output for both models. Among contributions greater than a
documented numerical tolerance, select the largest positive SHAP value. The
caption should be Top risk-increasing contributor, rather than an ambiguous Top
feature. Apply this definition even when the final prediction is legitimate.

If there is no positive contribution, show No risk-increasing contributor. If
SHAP is pending or failed, show Not computed / Unavailable instead; these states
are not equivalent. Resolve ties deterministically using stored feature order,
and identify ties in detailed data if needed.

Keep the strongest negative contribution in the narrative/details, so legitimate
predictions are not represented solely by the factors that increase their scores.
Largest positive contribution does not mean the largest total influence, a
globally important feature, or proof of fraud.

### Deterministic narrative (agreed)

- Generate text in the backend from the same contribution object used by the
  chart. No generative-language-model dependency is needed.
- Select up to two strongest positive and two strongest negative contributions.
- Map technical names to readable labels and correctly express actual values,
  including absence (for example, Not a merchant payment for type_PAYMENT=0).
- Use raw/unscaled readable values in prose; preserve exact model values in details.
- Do not invent high amount / unusual time thresholds, balance changes, account
  histories, or causal claims. Amount may influence via log_amount; label it as
  Transaction amount (log-transformed) in technical details.
- Do not imply that named top features alone fully account for the score; remaining
  contributions stay in the chart/data.
- Do not use the actual label to explain the prediction. Explain each model
  separately; never copy one model's explanation into the other's result.

Conditional example (only when those signs are actually observed):

> The transaction amount was the strongest factor increasing this model's fraud
> score. The transaction type lowered the score. The final score remained below
> the classification threshold, so the model predicted legitimate.

### SHAP details (agreed presentation)

Provide one model-specific waterfall graph with readable feature labels/values,
signed contributions, baseline, final output, and units. Identify the model and
record; include positive and negative effects and account for grouped remaining
features. Supply an accessible text description alongside the graphic. Explanation
version and raw contributions remain internal/export provenance, not a required
on-screen version panel. Numerical table and a separate raw-contribution download
are suggestions only; neither is required for View SHAP details.

Prefer fraud-probability-space explanations so baseline plus all contributions
reconstructs the displayed underlying score. Contributions displayed on a 0..100
scale are percentage-point changes in model output, not relative percentages or
proven changes in real-world fraud probability.

Proposed background: a fixed seeded sample of 200 ORIGINAL training records,
shared across both explainers, with interventional TreeSHAP in probability space.
Save sampled identities, seed, SHAP version, and configuration. This is a starting
proposal subject to a runtime check before it is frozen; it is not an approved
method or a guarantee of speed. Validate fraud-class selection and additivity.
The models can have different baseline outputs even with the same background.

Coverage proposal: compute explanations for loaded table pages and selected
details, then cache by run/record/explainer version. A full SHAP export is a separate
explicit job; mark partial/not-computed coverage honestly. The UI requirement to
show contributors for legitimate predictions expands explanation coverage beyond
the manuscript's emphasis on fraud predictions and should be documented.

## 6. Table and filters

Agreed: risk scores have no sorting controls; PaySim Transaction ID is a dataset
record ID; provide horizontal scrolling. Keep stable ordering by source record.

### Model identification (agreed implementation)

Retain the existing color-coded visual association for scores and contributors.
Do not add per-cell model badges or model subheaders in this implementation.
Keep consistent model/color mapping across screens. Existing model names elsewhere
in the table remain. Accessible programmatic names can identify model/value without
altering the visible design. Tooltips remain an optional suggestion, not a new
requirement. Removing this requirement does not remove model keys from backend data.

Suggestion only (not implementation plan): visible model badges/subheaders and
focus/touch-accessible tooltips would make model association clearer for people
who cannot distinguish the colors. The researchers chose to retain their design.

### Replacement filters (agreed)

- **Model:** Both (default), RF-SMOTE, Benchmark RF.
- **Prediction outcome:** All outcomes (default), True positive - fraud detected,
  False positive - false alarm, True negative - legitimate recognized,
  False negative - fraud missed.
- Disable Prediction outcome for Both, with helper text Select one model to filter
  by prediction outcome. Reset to All outcomes when switching back to Both.
- Disable correctness-based filtering when labels are unavailable. Never treat
  missing labels as legitimate or incorrect.
- Model selection determines displayed model columns and the outcome-filter
  reference. It does not select a different source dataset: both models process
  the same records. Selecting one model with All outcomes keeps all valid rows.
- Search and filters apply to the full result dataset, before pagination. Reset
  to the first page after changing search/filter; preserve state after details.
- Keep evaluation cards fixed to the declared evaluation population. Table filters
  do not silently recalculate official metrics or McNemar results.
- Optional later control for Both: Models disagree. This concerns predicted-label
  disagreement and can operate without ground truth; it is not a TP/FP filter.

Disabling the second filter is a chosen usability policy, not a mathematical
necessity: combined rules could be defined, but are unnecessary for this version.

### Table behavior (agreed, with model-label exception above)

Use semantic headers/caption, sticky header and record-ID column where feasible,
visible horizontal overflow, readable column widths, and accessible actions.
Preserve textual prediction labels; model identity within score/contributor cells
continues to use the chosen visual color mapping. Paginate on the server/file-query layer for large datasets;
do not load the entire dataset into the browser merely to display one page.
Show loading/empty/error states and matching versus total counts. A View details
button should work by keyboard and preserve source identity through every filter.

## 7. Individual researcher record

Agreed: transaction detail view, ground truth, return to prior table state.

Agreed layout and implementation guidance:

- **Original Inputs** (default): readable original amount/type, source step,
  source entity IDs, record ID, and independently supplied actual label. Display
  simulation time as simulated time, not an invented calendar timestamp.
- **Derived Inputs** (expandable): derived features before scaling and actual scaled
  values supplied to the classifier. Show both when explaining transformed values.
- **Source data** (optional expansion/export): all uploaded original columns. Mark
  balances and isFlaggedFraud as Not used by the model if displayed.
- **Results:** prediction, score, explanation, and researcher ground truth. Do not
  add a separate Correct / Incorrect or TP / FP / TN / FN badge to record details.
- No on-screen Model Run / Model information section for now. Preserve provenance
  internally and include it in downloads under the approved export placement.

The following remains the internal evaluation/filter definition, not an extra
record-detail display:

| Ground truth | Prediction | Outcome | Correctness |
| --- | --- | --- | --- |
| Fraud | Fraud | True positive | Correct |
| Legitimate | Fraud | False positive | Incorrect |
| Legitimate | Legitimate | True negative | Correct |
| Fraud | Legitimate | False negative | Incorrect |

Ordinary receipts lack independently established labels: omit correctness/outcome
and ground truth. Never ask users to supply a label as a predictor.

A model run is the saved training bundle used for the prediction, not the uploaded
file or the upload time. Retain its exact internal identifier and export it.
An analysis ID separately identifies one uploaded-file evaluation.
Do not hard-code a baseline run as the final model; later validated runs differ.

Suggestion only (not implementation plan): an expandable on-screen model-run
section would aid inspection, but has been deferred by the researchers.

## 8. Approved communication bands and prediction wording

Agreed wording: Predicted Fraud / Predicted Legitimate.

Display: Model risk score, out of 100, calculated as 100 * class-1 score.

| Unrounded score s | Approved band |
| --- | --- |
| 0 <= s < 20 | Minimal Risk |
| 20 <= s < 40 | Low Risk |
| 40 <= s < 60 | Moderate Risk |
| 60 <= s < 80 | High Risk |
| 80 <= s <= 100 | Critical Risk |

These are approved communication bands, not empirically validated fraud-risk
or operational intervention thresholds. The user's integer descriptions 20-39,
40-59, and 60-79 are represented as continuous half-open intervals so fractional
scores have no gaps. Critical Risk is the chosen communication label; it does not
establish verified fraud or a validated emergency-response rule.

Determine bands from the unrounded score; choose sufficient display precision or
boundary-aware formatting so rounding does not appear to contradict the band.
Position the marker from the score, and use one policy for all text and charts.

Prediction continues to use the saved shared classification threshold. The current
baseline is 0.50; the approved later validation procedure may select a different
shared threshold. Do not replace it with band boundaries. Moderate risk can fall
on either side of the decision threshold; explain this in a tooltip and show the
threshold marker in details. Bands do not claim calibrated real-world likelihood.

## 9. Evaluation information in downloads (agreed placement)

Add a Download results button to the researcher results page. Include the following
in the exported evaluation report/metadata instead of a new on-screen panel:

- Dataset/file and analysis ID.
- Evaluation scope: held-out test / validation / exploratory upload.
- Evaluated record count and actual class distribution.
- Model run and evaluation timestamp (distinct from training timestamp).
- Shared decision threshold.
- Whether same-record pairing and test membership were verified.

Exported technical details: dataset/split fingerprints, preprocessing version,
feature schema, RF/SMOTE settings, PR-AUC method, and SHAP configuration/coverage.

This information makes the result interpretable and reproducible. It does not
require additional manual user inputs or a database. Uploaded filenames and labels
alone cannot establish held-out status. Unknown provenance means exploratory,
not verified official thesis evaluation.

Suggestion only (not implementation plan): an on-screen evaluation-details panel
could make provenance easier to inspect. The researchers chose export-only placement.
Result titles must still avoid falsely identifying arbitrary uploads as official
test results; no new metadata panel is needed to prevent that misrepresentation.

## 10. Agreed statistical presentation and remaining method choices

- Rename Benchmark SMOTE Model to Benchmark RF (without SMOTE).
- Label confusion-matrix axes Actual and Predicted, with fraud as the positive class.
- Display all five metrics with definitions; display MCC as a coefficient -1..1.
- Signed percentage difference approved on 2026-09-22:
  Signed percentage difference = 100*(S-B)/((S+B)/2), for nonnegative metric scores
  with a positive denominator; S=RF-SMOTE, B=benchmark RF. Its magnitude matches
  the existing symmetric absolute difference, and its sign identifies direction.
  Positive means RF-SMOTE is higher; negative means benchmark RF is higher; zero
  means equal values when the denominator is positive. This is not baseline-relative
  percentage change. Align the manuscript/configuration/export/modal and sign guide.
  Active configuration now selects signed_over_mean; schema loading still accepts
  legacy absolute_over_mean. Calculation is pending Phase 4. Preserve old saved
  experiment configurations and identify the formula used in each export.
  Do not call descriptive differences statistically significant.
- If nonnegative metrics are both zero, show N/A - zero denominator. For negative
  MCC values, show the direct signed coefficient difference instead of implying a
  meaningful symmetric percentage; label its units and rationale. These approved
  presentation policies must be aligned with the manuscript and implementation.
  Do not use a negative or zero mean denominator to imply direction for MCC.
- Show N/A with a reason for mathematically undefined metrics. Do not substitute
  an unexplained zero or use an undefined metric in a difference calculation.
- Identify the finalized PR-AUC integration method and retain score-based inputs.
  Add a precision-recall curve if useful. AP versus trapezoidal integration remains
  open; do not relabel one as the other without documenting the choice.
- Retain the already-present McNemar table. Add method, statistic (correctly named
  for that method), alpha, evaluated sample size, and explicit discordant counts.
- The manuscript names continuity-corrected McNemar. Finalize small-discordance
  handling before implementing. Approved zero-discordance display: No discordant
  pairs; no evidence of a difference; p=1 by stated convention, statistic N/A.
  Any exact-test fallback must be documented, not silently substituted.
- Significant-result wording: Reject the null hypothesis. The models have different
  classification error rates on this evaluation set. Optionally state which had
  fewer errors, determined from paired predictions, without claiming every metric improved.
- Nonsignificant wording: Fail to reject the null hypothesis. There is insufficient
  evidence of different classification error rates on this evaluation set.
- Neutral p-value styling: a small p-value is not inherently a favorable result.
- Keep conclusions scoped to this population and selected threshold.

Agreed help interaction: clickable Precision, Recall, F1-Score, MCC, PR-AUC,
Percentage Difference, McNemar's Test, Contingency Table, and P-Value labels open
accessible modals. Copy is in [UI_HELP_CONTENT.md](UI_HELP_CONTENT.md). Formula
variants must follow actual computation metadata; draft content does not select
an unresolved statistical method.

## 11. Proposed landing-page copy

This is draft product copy for the planned feature-complete prototype. Before
publishing it, gate unavailable actions or label them Coming soon; do not claim
OCR, explanations, exports, or evaluation are operational until connected.

### Hero

**INTEGRITREE**

**Understand transaction predictions, scores, and the features behind them.**

Explore an explainable e-wallet fraud detection research prototype. Compare
Random Forest models trained with and without SMOTE on synthetic PaySim data,
then inspect their predictions and SHAP-based explanations.

Primary action: **Analyze a dataset**

Secondary action: **Try the receipt demonstration**

Supporting line: **Built for research evaluation and experimental exploration.**

### What you can explore

**Compare two models**

See how benchmark Random Forest and RF-SMOTE classify the same transaction records.
Review agreements, disagreements, and performance against available ground truth.

**Review model risk scores**

View each model's fraud score alongside its predicted classification and the
threshold used to make that decision.

**Understand influential features**

Identify the strongest risk-increasing contributor, read a short explanation,
and open a detailed SHAP view of features that raise or lower the score.

**Evaluate research results**

Inspect confusion matrices, precision, recall, F1-score, MCC, PR-AUC, descriptive
model differences, and McNemar's test on compatible labeled data.

### How it works - For researchers

1. **Upload compatible transaction data.** Download the CSV template or upload a
   compatible PaySim file. Ground-truth labels are required for evaluation.
2. **Review validation.** Check the file preview and resolve missing or invalid
   inputs before analysis proceeds.
3. **Analyze with both models.** Integritree applies saved preprocessing and runs
   benchmark RF and RF-SMOTE on the same valid records.
4. **Explore individual results.** Search records, filter prediction outcomes, and
   compare each model's classification, score, and explanation.
5. **Review model performance.** Inspect metrics and statistical comparison for the
   clearly identified evaluation set.
6. **Export your findings.** Download results with the model and evaluation details
   needed to trace and reproduce the analysis.

### How it works - For the receipt demonstration

1. **Upload a supported receipt.** Choose a GCash person-to-person transfer receipt
   in a supported image format.
2. **Confirm transaction details.** Review extracted fields, correct mistakes, and
   complete required information.
3. **Check compatibility.** Integritree checks whether the confirmed details fit
   the supported demonstration workflow.
4. **Compare model outputs.** Review both models' predicted classifications and
   model risk scores. The models may disagree.
5. **Explore the explanation.** Read the main influences and open SHAP details to
   inspect contributions that increase or decrease each score.

### How the research models are prepared

Both models use the same feature definitions and shared validation/test records.
Benchmark RF learns from the original imbalanced training set. RF-SMOTE learns
from a training set augmented with synthetic minority-class examples. Training
happens separately from uploaded-file analysis; the application uses saved models.

### What the results mean

Integritree evaluates models using synthetic PaySim transactions. Its predictions
are model outputs, not verified determinations of fraud. The receipt workflow is
an experimental demonstration whose real-world GCash performance has not been
established. It does not verify receipt authenticity or identify image editing.

### Short FAQ

**Does a fraud prediction prove that a transaction is fraudulent?**

No. It means the selected model's fraud score reached its classification threshold.

**Why can the models disagree?**

They learn from different training distributions, so the same record can receive
different scores and classifications.

**What does SHAP explain?**

It shows how transaction features move a model's output above or below its baseline.
It explains the prediction rather than proving the cause of actual fraud.

**Can a receipt upload produce evaluation metrics?**

A receipt alone does not supply independently verified fraud labels. Evaluation
metrics are available for compatible labeled research data.

**Does Integritree verify whether a receipt is genuine?**

No. Receipt images supply transaction details for the experimental workflow.

### Team and closing action

Keep the existing team names/cards; do not invent roles or affiliations.

**Start exploring transaction results.**

Choose dataset analysis for research evaluation or the receipt demonstration for
individual experimental predictions.

Actions: **Analyze a dataset** / **Try the receipt demonstration**

## 12. Agreed supporting requirements and proposed implementation details

### Exports

Agreed: Download results button with transaction results, evaluation, and relevant
provenance. No separate on-screen evaluation/run-information panel.

Approved file type: a single ZIP download containing:

- results.csv: per-record identity, both predictions/scores/bands, actual label when
  available, and top contributors with explanation status. No need for duplicate
  correctness badges in the UI; internal/export outcome codes are optional.
- evaluation.json: exact metrics, confusion matrices, paired McNemar table and test,
  method identifiers, threshold, input/evaluation population, and edge-case statuses.
- report.html: a readable offline report with embedded styling, evaluation context,
  model/run provenance, results summary, limitations, and a data dictionary. No
  external assets required; escape untrusted input. This is not a full-million-row
  HTML table. No export renderer has been implemented in this documentation update.
- metadata.json: analysis ID, model/preprocessor versions, dataset/split identity,
  selected methods, timestamps, export scope, and SHAP coverage.
- shap_values.csv: optional available contributions; explicitly state subset coverage.

A summary XLSX/PDF remains an optional future suggestion, outside this approved
ZIP scope. A single Excel worksheet cannot hold the full PaySim
dataset (Excel has a 1,048,576-row worksheet limit). CSV itself can exceed that limit;
opening a large CSV in Excel does not remove Excel's limit. Do not silently truncate.

Detailed SHAP values may require a separate export job. Distinguish
Export all results from Export filtered records. Mark filtered exports as subsets.
Use standard CSV escaping and protect spreadsheet readers against formula-like
untrusted text. Never turn an unavailable value into an invented zero.

### States, privacy, and access

Agreed: progress, recovery, empty/error states, analyze-another/replace/review-input
actions, accessible/responsive UI, and explicit receipt handling.

Proposed progress stages: uploading, validating or extracting, awaiting confirmation,
predicting, explaining, ready, partial failure, failed. Use real measured progress
where available; otherwise show an indeterminate stage rather than invented progress.

Proposed receipt policy: temporary processing, no automatic permanent receipt
history, and explicit cleanup of uploads and personal details. Retention duration,
expiration, and recovery behavior remain open. Keep research artifact preservation
separate from personal receipt retention. Isolate analysis access if multiple users
can reach the service; do not expose files/results by a guessable record ID alone.

An application database remains optional. Files plus run metadata can support the
prototype; accounts/history/persistent shared jobs would require a further design.

### Frontend touchpoints

- pages/LandingPage.jsx and .css: draft copy, two entry paths, scope, FAQ.
- pages/UploadPage.jsx and .css: schema/template, preview, validation, supported scope.
- New/extended confirmation view: editable structured receipt fields and provenance.
- components/TransactionTable.jsx and .css: approved color mapping, filters, scroll, page retrieval,
  detail navigation, explicit explanation states, no risk sorting.
- pages/ResearcherResultsPage.jsx and .css: statistics, help modals, Download results,
  Original Inputs / Derived Inputs, ground truth, SHAP views. No new evaluation
  panel, model-run section, or correctness badge.
- pages/UserResultsPage.jsx and .css: confirmed fields, both model outputs,
  experimental interpretation, SHAP, no fabricated ground truth/correctness.
- Shared score, prediction-label, and explanation components: one presentation policy.

### Backend touchpoints

- Receipt ingestion/OCR/adapter and separate validated input contract.
- Shared preprocessing refactor preserving raw PaySim behavior and saved scaling.
- Evaluation service: paired rows, threshold, metric conventions, McNemar, metadata.
- Explanation service: target class, background, values, narrative, plots/data, cache.
- Research query API: search/filter before pagination; stable identity/order.
- Exports and job states; upload cleanup and access isolation as needed.

These are implementation targets, not claims that the endpoints/services exist.

## Acceptance checks for later implementation

- Accepted CSV templates match the actual ingestion contract.
- Original and derived-input paths produce identical model matrices for equivalent
  PaySim data; saved scaling occurs once and is never refitted on uploads.
- Receipt inputs obey only the agreed mapping, reject unsupported cases, and carry
  experimental provenance. No generated fake source IDs or step assumptions.
- Every prediction/explanation stays joined to its original record and model run.
- SHAP values reconstruct fraud output within the recorded tolerance; narrative,
  top-positive selection, zero/no-positive cases, and waterfall agree.
- Both-model, legitimate, disagreement, missing-label, and failed-SHAP states work.
- Outcomes match the four ground-truth/prediction cases; Both resets the outcome filter.
- Search/filter apply across all rows; paging and return navigation preserve state.
- Table filters do not change the official evaluation population silently.
- Band boundaries, threshold ties, rounding, and labels follow the selected policy.
- Statistical method/edge cases are approved before producing official results.
- Exports preserve identities, provenance, precision, and honest coverage/missing values.
- Keyboard, touch, small-screen, upload failure, and cleanup behavior are checked.

## Agreed Research Scope and Limitations page

Use the title Research Scope and Limitations.
Explain PaySim-only evaluation, experimental receipt mappings, unknown calendar
anchor/currency equivalence, supported input types, prediction uncertainty,
communication bands, SHAP's non-causal meaning, and lack of receipt-authenticity
verification. Keep privacy/retention promises tied to the actual implementation.
See [page copy](UI_HELP_CONTENT.md#research-scope-and-limitations-page).

Link the page from the footer, receipt upload/confirmation, results, and export.
Keep one short contextual statement near receipt submission/results: a separate
page does not itself validate the mapping or ensure the user sees its limitations.
Do not add a mandatory acceptance checkbox unless separately requested/justified.

## Implementation details still to finalize

The mock-up corrections can proceed. The following details do not reopen approved
presentation decisions; resolve each before its dependent backend feature ships.

1. Entity dropdown presentation and optional name/reference handling. Receipt scope
   is settled: person-to-person only, with the other four type options disabled.
   Exact names remain optional in the recommendation; an unreadable reference is proposed
   to remain unavailable while an internal analysis ID preserves traceability.
2. Exact weekday/time convention and documented numeric amount mapping, following
   acknowledgement of their limitations. No conversion validated by these decisions.
3. Final SHAP background size/seed/runtime coverage and export-job lifecycle.
4. Optional model-association tooltips; visible model badges remain suggestions only.
5. PR-AUC integration method and McNemar small-discordance handling. The signed
   comparison formula, negative-MCC comparison, undefined-value display, and
   zero-discordance presentation are approved.
6. Landing-page draft copy and retention duration. Help/limitations content is
   approved, subject to filling in actual methods and implemented retention facts.

Approved and no longer open: communication-band names/boundaries; Original Inputs
and Derived Inputs names; no correctness badges; no on-screen model-run/evaluation
panel; color-only visible model association in score/contributor cells; SHAP
three-level presentation and deterministic narrative; evaluation metadata in exports;
person-to-person receipt scope with a five-option, TRANSFER-only-enabled dropdown;
replacement filters; ZIP export; signed symmetric percentage difference and its
edge-case presentation; standardized help modals; Research Scope and Limitations page.

## References

- [Thesis context and prior decisions](THESIS_CONTEXT.md)
- [Methodology and approved model configuration](METHODOLOGY.md)
- [Current API boundaries](API.md)
- [SHAP TreeExplainer](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html)
- [SHAP waterfall plot](https://shap.readthedocs.io/en/latest/generated/shap.plots.waterfall.html)
- [W3C accessible tables](https://www.w3.org/WAI/tutorials/tables/)
- [McNemar implementation](https://www.statsmodels.org/stable/generated/statsmodels.stats.contingency_tables.mcnemar.html)
- [Average precision convention](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)
- [Probability calibration](https://scikit-learn.org/stable/modules/calibration.html)
- [UI help and limitations draft](UI_HELP_CONTENT.md)
- [PaySim entity-code analysis](https://repository.tudelft.nl/file/File_dcf25184-4f67-4471-94fd-b3a1935073f9?preview=1)
- [Excel worksheet limits](https://support.microsoft.com/en-gb/excel/excel-specifications-and-limits)

## Change history

- 2026-09-22 (backend alignment review): Deferred the numerical SHAP table; waterfall
  is the sole expanded SHAP view. Audited Phases 1-3, aligned the active signed
  comparison configuration with legacy compatibility, and updated future-phase
  requirements. See BACKEND_ALIGNMENT_AUDIT.md for evidence and limits.
- 2026-09-22 (latest approval): Confirmed frontend ownership of SHAP numerical-table
  formatting. Recorded approvals for items 6-9, signed percentage difference and
  other statistical/help recommendations. Accepted the ZIP package, replacement
  filters, help content, and limitations page. Unselected methodological alternatives
  remain open. Updated documentation only; application/configuration migration remains.
- 2026-09-22: Recorded approved SHAP presentation, color-only model association,
  input section names, removal of correctness/run UI, communication bands, and
  export-only metadata. Reopened percentage convention for a signed option. Added
  proposed receipt dropdowns/optional masked names, ZIP recommendation, standardized
  help copy, and limitations-page proposal. Application code remains unchanged.
  Follow-up confirmed person-to-person scope and visible disabled non-transfer options.
- 2026-09-21: Initial record from the researchers' numbered response. Recorded
  explicit agreements separately from proposed resolutions. Included corrected
  results-page coverage, ignored screenshot numbers, and drafted landing copy.
