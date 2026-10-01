# Confirmed receipt mapping

Implemented 2026-10-01: internal confirmation contracts, versioned feature mapping,
shared saved scaling, and paired prediction. **Image jobs, receipt HTTP routes,
frontend confirmation, SHAP/export integration and cleanup are now connected** in
the [receipt application](RECEIPT_APPLICATION.md).
This milestone does not enable an image-free manual-entry workflow.

## Supported mapping boundary

Mapping version: `gcash_confirmed_v2`. Confirmed origin and destination roles are
`client` or `merchant`, independently editable. Both merchant flags use these values.
The form defaults are:

| Category | Origin | Destination |
| --- | --- | --- |
| PAYMENT | Client | Merchant |
| TRANSFER | Client | Client |
| DEBIT | Client | Merchant |
| CASH_IN | Merchant | Client |
| CASH_OUT | Client | Merchant |

Changing category resets both roles to these defaults; users may then override them.
Express Send initially selects TRANSFER, Pay Online PAYMENT, and Bank Transfer DEBIT.
All five categories may be confirmed for a supported uploaded screenshot. This does
not enable new Cash In/Out or QR screenshot layouts. Layout recognition remains unchanged.
The detected workflow must still match the server-owned layout, but confirmed category
and model roles may differ from OCR candidates and observed source roles. Source
observations remain immutable evidence, not additional model-role options.
Wallet-funding assertions are no longer required. Proceed is explicit confirmation.
These are user-selected demonstration assumptions, not validated GCash/PaySim equivalence.

## Internal contracts

`backend/src/integritree/receipts/contracts.py` defines:

- `ExtractedReceiptFields`: optional OCR candidate text, including invalid or
  missing values, retained separately from the user's confirmed values.
- `ReceiptSource`: server-owned analysis UUID, image UUID, SHA256, PNG/JPEG MIME
  type, accepted layout ID, extractor/parser versions, candidates, and an optional
  observed destination role. UUIDs are separate from the payment reference.
- `ConfirmedReceiptFields`: strict required workflow, category, principal, date,
  time and origin/destination roles; optional string reference and
  names. Currency is fixed to PHP and timezone to Asia/Manila.
- `ConfirmedReceipt`: validated source and fields, explicit boolean confirmation,
  positive input revision, and mapping version. Models are frozen and revalidated
  when they enter the mapping, including instances changed using `model_copy`.

**ReceiptSource is trusted service context, not a client request body.** Its initial
layout IDs (`gcash_express_send_v1`, `gcash_pay_online_v1`,
`gcash_bank_transfer_v1`) identify mapping targets. The image service verifies
upload ownership/existence/fingerprint, file limits and actual supported layout
before creating this context. The current mapping validates metadata consistency;
it does not read or authenticate an image. A parser's workflow guess cannot be used
as layout authorization. Known destination-role evidence must be retained in the
source context rather than overwritten by the user's selection. An unresolved
layout must stop before this boundary.

Example confirmed field payload, to be validated only within that upload context:

```json
{
  "workflow": "pay_online",
  "category": "PAYMENT",
  "amount": "125.50",
  "date": "2026-10-05",
  "time": "14:30",
  "currency": "PHP",
  "timezone": "Asia/Manila",
  "origin_role": "client",
  "destination_role": "merchant",
  "reference": "00001234"
}
```

Principal is an unsigned plain decimal string with at most two decimal places
(or a Python Decimal internally). Zero is allowed. Binary floats, grouped/currency
text, exponents, negative/nonfinite values and values that change on decimal-to-model
float roundtrip are rejected. The string has a 32-character guard. No fee/total
substitution or amount imputation occurs. The UI must submit the confirmed principal
in this canonical format; JSON output keeps two decimal places.

Date is a valid `YYYY-MM-DD`; time is unambiguous 24-hour `HH:MM` or `HH:MM:SS`.
Seconds are retained but the hour feature uses the hour component. Offset-bearing
times and other timezones are rejected rather than silently converted. Optional
reference/names accept nonblank strings up to 256 characters, or null when unavailable;
reference leading zeros are preserved. Extra properties, including labels, balances,
raw PaySim fields and supplied feature vectors, are rejected. Validation errors
identify fields or compatibility rules; the HTTP handler omits raw input
and exception context from public errors/logs.

## Confirmation, provenance and prediction

`confirm_receipt(source, fields, confirmed=True, previous=...)` creates revision 1
or advances an existing confirmation of the same source. Even reconfirmation creates
a new revision. It requires an actual boolean true, not `1` or a string. The
confirmed record retains the source/candidates and exposes per-field extracted and
confirmed values with `unchanged`, `completed`, `corrected`, or `unavailable` status.
Comparison is literal, so normalization such as `100` to `100.00` is recorded as a
correction. Currency and timezone are explicit mapping facts,
not inferred OCR evidence.

`receipt_features(receipt)` constructs exactly the existing eleven columns in saved
order. Date supplies Monday=0 weekday; local time supplies hour 0..23; category
supplies one-hot indicators; principal supplies `log1p(amount)` and the zero flag;
confirmed roles supply merchant flags. No simulation step or fake C/M account ID is
constructed. Names, references, image metadata and ground truth never enter features.

`FittedPreprocessor.transform_engineered` applies the saved multiply-then-add
scaling once to internally generated features. Raw PaySim `transform` now delegates
to the same method after its existing feature engineering. State schema, feature
order, missing raw amount/type behavior, absence of clipping and saved parameters
are unchanged. This method is low-level code, not an arbitrary feature-matrix API.

`predict_receipt(bundle, receipt)` uses a trusted loaded bundle, both models and its
saved shared threshold. The internal result contains the confirmation snapshot,
unscaled/scaled features and paired predictions with analysis ID and model run ID.
It produces no ground truth or evaluation metrics. The application service reuses
this validated path before computing receipt SHAP and building ZIPs.

`input_sha256` binds the image context, extracted/confirmed fields, revision and
mapping version. `ReceiptPrediction.is_current(receipt)` rejects results from a
different confirmation. The application service stores this identity and
invalidates cached predictions, explanations and downloads on revision. The helper
does not itself manage application sessions or delete cached data.

## Verification

Synthetic tests in `backend/tests/test_receipt_mapping.py` check all 168 weekday/hour
combinations for each of the three mappings against equivalent raw PaySim features.
Test-only source steps and entity IDs serve as equivalence witnesses; production
receipt code never fabricates them. Tests compare saved/reloaded scaling and paired
predictions, zero/fractional/out-of-training-range amounts, optional references,
all five categories and four role combinations, layout binding, strict input rejection, provenance and stale result detection.
At the internal mapping milestone, the backend suite passed **293 tests**, with five dependency deprecation
warnings. Saved research models, data, threshold and dependency lock remain unchanged.

This is software parity, not evidence that real GCash fraud detection is validated.
See [receipt scope](RECEIPT_WORKFLOW.md) and [OCR baseline limits](RECEIPT_OCR_BENCHMARK.md).
