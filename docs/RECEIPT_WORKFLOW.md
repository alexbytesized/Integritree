# Receipt workflow decisions and sample collection

Approved scope updated 2026-09-30. **Receipt processing is planned, not implemented.**
The OCR adapter, parser, confirmed-input mapping, and receipt prediction routes
remain scaffolds. This decision supersedes the earlier person-to-person-only
restriction; it does not change model training, saved artifacts, or evaluation.

## Initial five-category scope

Accept screenshots of completed transaction receipts/details from the GCash app.
Start with one workflow per PaySim category:

| PaySim category | Initial GCash workflow | Interpretation |
| --- | --- | --- |
| TRANSFER | Express Send between GCash users | Mobile-wallet user to another mobile-wallet user. |
| CASH_IN | Over-the-counter cash-in into GCash | Cash deposited into a wallet through an agent/partner. |
| CASH_OUT | Over-the-counter cash-out | Wallet funds withdrawn as cash through an agent/partner. |
| PAYMENT | Merchant QR payment funded from the GCash wallet | Payment for goods/services to a merchant. |
| DEBIT | GCash Bank Transfer to a bank account | Money moved from the wallet to a bank account. |

This is an approved experimental application mapping, not evidence that GCash and
PaySim have equivalent fraud patterns. PaySim DEBIT denotes wallet-to-bank movement;
it does not mean every transaction that decreases a wallet balance. A QR code alone
does not distinguish a personal transfer from a merchant payment. Keep the original
GCash workflow label alongside the mapped category and resolve ambiguous inputs
before prediction; do not silently force them into a category.

Bank-funded cash-in, ATM cash-out, bills, load, online payments, SMS/email screenshots,
and photos of printed receipts are outside this initial scope. Exact supported app
layouts must be established from samples. If the app provides no usable screenshot
for a selected workflow, report the gap rather than substituting another source.
All five categories are implementation targets, not five operational receipt routes.

## Settled behavior awaiting implementation

- Run OCR locally on the backend computer; do not send images to an external OCR
  service. The engine and dependency versions remain to be selected.
- Require an uploaded supported screenshot. Let users correct extracted fields
  and complete unreadable/missing fields before explicit confirmation. A separate
  image-free manual-entry workflow is not included. Manual completion does not
  bypass unsupported workflow or unresolved account-role validation.
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
- Validate account roles for each workflow. The earlier requirement that both
  parties be personal accounts no longer covers the full scope. Mapping banks,
  agents, and merchants into the model's entity indicators remains unresolved.
  Do not infer C/M indicators from names or automatically equate an agent with a
  PaySim merchant. Unknown or incompatible required roles must block prediction.
- Use a separate confirmed-receipt contract, derive the same eleven unscaled
  features, and apply saved scaling exactly once. Do not fabricate a PaySim step
  or account ID, accept arbitrary client matrices, or refit preprocessing.
  Version the mapping and preserve extracted/confirmed values and correction
  provenance; changed inputs invalidate prior predictions and explanations.
- Return both saved models' predictions, scores, and explanations without invented
  ground truth or evaluation metrics. Keep the saved shared classification threshold.
- Keep application images, confirmed details, and results temporarily until Clear
  or backend shutdown/restart; no permanent receipt history or timed expiry is
  selected. Closing a browser tab alone does not promise deletion. Implement access
  isolation and cleanup before publishing these behaviors as working features.
  Exclude personal receipt contents from logs and committed fixtures.

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

Use neutral names such as `transfer_001.png` or `debit_001.png`. Put uncertain
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

1. Inspect representative screenshots and record exact supported layouts/gaps.
2. Resolve and version account-role/merchant-indicator mappings for all workflows.
3. Select the local OCR engine and parser approach; finalize upload limits, receipt
   job/API contracts, error behavior, and any receipt-result download requirements.
4. Implement and verify the confirmed-input prediction path, preserving raw PaySim
   inference and shared-scaler parity, then connect image extraction and the frontend.
5. Verify every supported workflow, ambiguous category/roles, missing references,
   fee/principal separation, corrections, failures, cleanup, and no fabricated labels.

No retraining is required solely to expose existing transaction categories.
Changing feature semantics requires a separate methodology decision. Official
held-out evaluation still follows [the researcher workflow](RESEARCHER_WORKFLOW.md).

## Definition references

- [PaySim paper: transaction definitions](https://www.researchgate.net/publication/313138956_PAYSIM_A_FINANCIAL_MOBILE_MONEY_SIMULATOR_FOR_FRAUD_DETECTION)
- [GCash: Send Money, Bank Transfer, Cash In, Cash Out](https://help.gcash.com/hc/en-us/articles/40136231787929-Differences-between-Send-Money-Bank-Transfer-Cash-In-Cash-Out)
- [GCash: merchant QR payments](https://help.gcash.com/hc/en-us/articles/360017563034-How-to-scan-to-pay-with-my-QR)
