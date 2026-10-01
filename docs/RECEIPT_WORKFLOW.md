# Receipt workflow decisions and sample collection

Updated 2026-10-01. **Receipt application processing is connected for Express Send,
Pay Online and confirmed bank-account transfers.** RapidOCR is the provisional
integration choice after [comparison with Tesseract](RECEIPT_OCR_BENCHMARK.md).
The [internal mapping and paired prediction path](RECEIPT_MAPPING.md) is implemented
for Express Send, Pay Online and confirmed bank-account transfers. Image jobs,
receipt HTTP routes, confirmation UI, paired SHAP, ZIPs and cleanup are implemented;
see [application operation and contracts](RECEIPT_APPLICATION.md). The five-category
scope supersedes the earlier person-to-person-only restriction and does not change
model training, saved artifacts, or research evaluation.

## Initial five-category scope

Accept screenshots of completed transaction receipts/details from the GCash app.
Start with the selected workflows below; PAYMENT was expanded after sample review:

| PaySim category | Initial GCash workflow | Interpretation |
| --- | --- | --- |
| TRANSFER | Express Send between GCash users | Mobile-wallet user to another mobile-wallet user. |
| CASH_IN | Over-the-counter cash-in into GCash | Cash deposited into a wallet through an agent/partner. |
| CASH_OUT | Over-the-counter cash-out | Wallet funds withdrawn as cash through an agent/partner. |
| PAYMENT | Merchant QR and Pay Online payments funded from the GCash wallet | Payment for goods/services to a merchant. |
| DEBIT | GCash Bank Transfer to a bank account | Money moved from the wallet to a bank account. |

This is an approved experimental application mapping, not evidence that GCash and
PaySim have equivalent fraud patterns. PaySim DEBIT denotes wallet-to-bank movement;
it does not mean every transaction that decreases a wallet balance. A QR code alone
does not distinguish a personal transfer from a merchant payment. Keep the original
GCash workflow label alongside the mapped category and resolve ambiguous inputs
before prediction; do not silently force them into a category.

Bank-funded cash-in, ATM cash-out, bills, load, credit-funded payments, SMS/email screenshots,
and photos of printed receipts are outside this initial scope. Exact supported app
layouts must be established from samples. If the app provides no usable screenshot
for a selected workflow, report the gap rather than substituting another source.
All five categories are implementation targets, not five operational receipt routes.
Release verified workflows as they pass checks; do not wait for every category.
CASH_IN/CASH_OUT screenshot recognition remains pending representative samples. All five model categories are editable for a supported image using mapping v2.

## Settled behavior and implementation boundaries

- Run OCR locally on the backend computer; do not send images to an external OCR
  service. The completed [local comparison](RECEIPT_OCR_BENCHMARK.md) provisionally
  selects RapidOCR; checked date and account-role failures still require review.
  Keep OCR dependencies isolated from frozen model dependencies.
- Accept one PNG/JPEG screenshot at a time, at most 10 MiB of actual uploaded bytes,
  with a 20,000,000 decoded-pixel guard. Frontend and server byte limits are
  implemented; the server checks decoded dimensions before OCR.
- Require an uploaded supported screenshot. Let users correct extracted fields
  and complete unreadable/missing fields before explicit confirmation. A separate
  image-free manual-entry workflow is not included. Manual completion does not
  bypass the supported-image gate. Category and Client/Merchant roles are editable.
- Interpret confirmed date/time in Asia/Manila; derive hour 0..23 and weekday
  Monday=0 through Sunday=6. Preserve an unambiguous time. This is a demonstration
  convention, not a verified alignment with PaySim's simulation clock.
- Confirm the PHP transaction principal, separately from fees, balances, and totals.
  Parse amounts safely as decimals before model conversion. Use the numeric PHP
  principal for log1p(amount) and the zero-amount indicator, then reuse the saved
  transformations. No currency conversion or PaySim-unit equivalence is established.
- Receipt reference is optional. Preserve leading zeros and visible text as a
  string when available; otherwise mark it unavailable. Use a separate internal
  analysis ID. Optional masked names are traceability only; names, phone numbers,
  references, balances, and labels are not receipt model predictors.
- Mapping v2 uses editable Client/Merchant roles. Type changes apply defaults:
  PAYMENT, DEBIT and CASH_OUT Client-to-Merchant; TRANSFER Client-to-Client;
  CASH_IN Merchant-to-Client. Users may override both roles. Preserve detected
  workflow/layout/observed roles as source evidence; only confirmed category and
  model roles drive their corresponding features. No wallet-funding assertion is
  required. These user-selected assumptions supersede the original v1 restrictions.
- Use a separate confirmed-receipt contract, derive the same eleven unscaled
  features, and apply saved scaling exactly once. Do not fabricate a PaySim step
  or account ID, accept arbitrary client matrices, or refit preprocessing.
  Version the mapping and preserve extracted/confirmed values and correction
  provenance; changed inputs invalidate prior predictions and explanations.
- Return both saved models' predictions, scores, and explanations without invented
  ground truth or evaluation metrics. Keep the saved shared classification threshold.
- Provide an individual-analysis ZIP containing confirmed inputs, paired predictions,
  explanation results, and mapping/model provenance. Exclude the original image and
  evaluation metrics. Routes and exact files are documented in the application guide.
- Keep images, drafts and results only while the receipt flow is active. Leaving
  the flow or Clear invalidates access immediately; active workers finish before
  file deletion. Closing the final session tab clears after a 30-second reconnect
  grace period. Refresh and Results-to-edit navigation preserve the receipt.
  Backend shutdown/restart clears temporary data. See [session presence and network
  limits](RECEIPT_APPLICATION.md). Exclude private receipt contents from logs/fixtures.

## Collect development screenshots

Save local samples under this project-relative path (already ignored by Git):

```text
backend/data/raw/receipt_samples/
├── TRANSFER/
├── CASH_IN/
├── CASH_OUT/
├── PAYMENT/
├── DEBIT/
└── UNSURE/
```

Start with **3–5 screenshots per category**, where available (15–25 total).
This is a practical inspection target, not a training-set minimum, maximum, or
statistically sufficient OCR benchmark. Prefer distinct layouts, amounts, date/time
formats, and image sizes to repeated copies of the same example. Add difficult
examples and reserve some unseen screenshots for testing after extraction rules
are developed. Record missing categories/layouts instead of manufacturing evidence.

The current collection contains 123 TRANSFER, 19 PAYMENT, and 4 DEBIT images;
CASH_IN, CASH_OUT, and UNSURE are empty. These are collection labels, not verified
layout classifications. All 146 images were renamed without content changes to
four-digit names such as `transfer_0023.jpg` and `debit_0002.jpg`. The 3–5 example
target is not a cap; retain all usable originals and keep duplicate groups together
when separating development and verification examples.

Use neutral four-digit filenames. Put uncertain
examples in `UNSURE`; folder placement is a provisional collection label, not a
validated model input. Preserve labels, layout, amount, and date/time. Redacted
copies are recommended for shared inspection; real names, phone numbers, and
reference values are unnecessary. Own/consented unredacted examples may be used
privately for local testing. Redaction is not an application input requirement.
Use synthetic reference strings, including leading zeros, in committed fixtures.

These are **retained development samples**, separate from temporary application
uploads. Application cleanup must never delete this collection, raw PaySim data,
prepared data, models, or research reports. Do not force-add personal screenshots
to Git. On a fresh checkout, recreate the ignored folders with PowerShell from
the repository root:

```powershell
$receiptSampleRoot = Join-Path (Get-Location) 'backend/data/raw/receipt_samples'
'TRANSFER', 'CASH_IN', 'CASH_OUT', 'PAYMENT', 'DEBIT', 'UNSURE' | ForEach-Object {
    New-Item -ItemType Directory -Force -Path (Join-Path $receiptSampleRoot $_) | Out-Null
}
```

## Dataset coverage and limits

The dataset and both trained models support all five transaction types. The
[dataset audit recorded in the methodology](METHODOLOGY.md) distinguishes category
coverage from observed fraud labels:

| Category | Present in dataset/model inputs | Fraud-labeled examples in this dataset |
| --- | --- | --- |
| CASH_IN | Yes | None |
| CASH_OUT | Yes | Present |
| DEBIT | Yes | None |
| PAYMENT | Yes | None |
| TRANSFER | Yes | Present |

This is a limitation of the synthetic fraud scenarios and labels, not evidence
that the other categories cannot involve fraud in real life. There are no original
fraud-positive examples for CASH_IN, PAYMENT, or DEBIT from which to learn their
category-specific fraud patterns. SMOTE cannot supply missing real-world evidence;
fractional synthetic type indicators do not establish authentic category coverage.
Do not filter out the other types or claim validated fraud detection for every
GCash workflow. That would require representative independently labeled data and
separate evaluation. OCR sample receipts are not a fraud-model training dataset.

## Remaining decisions and next implementation sequence

1. Completed the initial audit/comparison: all 146 files decode; 10 duplicate pairs,
   including one cross-category conflict; 21 visually annotated examples. See the
   [benchmark and layout gaps](RECEIPT_OCR_BENCHMARK.md). Keep unresolved examples for
   review; merchant QR and cash-in/out remain unverified.
2. Completed [confirmed-input contracts and mapping](RECEIPT_MAPPING.md), with
   shared-scaler and paired prediction parity for Express Send/TRANSFER, Pay Online/
   PAYMENT and bank-account DEBIT. Known wallet-destination conflicts are rejected.
   Cash-agent mapping and merchant QR layout validation remain pending.
3. Completed [receipt HTTP jobs, confirmation, SHAP/ZIPs, frontend connection and
   cleanup](RECEIPT_APPLICATION.md). Revisions invalidate old predictions and assets.
4. Synthetic API/browser checks and one real Express Send smoke test pass. Collect
   additional unseen Pay Online and bank-account examples for acceptance checks;
   this smoke test is not an independent OCR accuracy estimate.
5. Collect cash-category and merchant QR samples, settle remaining layouts/roles,
   then extend their gates and mappings. Any parser tuning requires fresh unseen
   examples for independent verification.

No retraining is required solely to expose existing transaction categories.
Changing feature semantics requires a separate methodology decision. Official
held-out evaluation still follows [the researcher workflow](RESEARCHER_WORKFLOW.md).

## Definition references

- [PaySim paper: transaction definitions](https://www.researchgate.net/publication/313138956_PAYSIM_A_FINANCIAL_MOBILE_MONEY_SIMULATOR_FOR_FRAUD_DETECTION)
- [GCash: Send Money, Bank Transfer, Cash In, Cash Out](https://help.gcash.com/hc/en-us/articles/40136231787929-Differences-between-Send-Money-Bank-Transfer-Cash-In-Cash-Out)
- [GCash: merchant QR payments](https://help.gcash.com/hc/en-us/articles/360017563034-How-to-scan-to-pay-with-my-QR)
