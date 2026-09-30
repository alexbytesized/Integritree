# Local receipt OCR baseline

Measured 2026-10-01 on this Windows computer. **RapidOCR is the provisional engine
for integration**, selected from development results before verification. This is
an extraction baseline, not a released receipt feature or fraud-model evaluation.
No OCR fine-tuning, fraud-model retraining, or PaySim held-out evaluation was run.

## Collection audit

All 146 original JPEGs passed the 10 MiB byte and 20,000,000 decoded-pixel limits.
The inventory found 10 pixel-identical duplicate pairs, leaving 136 unique pixel
groups. Duplicates stay in one split; originals were not moved, deleted, or edited.
Near duplicates are not detected automatically and still need review.

| Folder label | Files | Development | Verification | Review only |
| --- | ---: | ---: | ---: | ---: |
| TRANSFER | 123 | 94 | 28 | 1 |
| PAYMENT | 19 | 14 | 4 | 1 |
| DEBIT | 4 | 2 | 2 | 0 |
| CASH_IN / CASH_OUT / UNSURE | 0 | 0 | 0 | 0 |
| Total | 146 | 110 | 34 | 2 |

`TRANSFER/transfer_0006.jpg` and `PAYMENT/payment_0014.jpg` are the same image
with conflicting folder labels. Both are excluded from development/verification
pending workflow review. Folder labels are collection labels, not verified truth.

The split is deterministic by decoded-pixel hash, stratified by folder, with
previously previewed examples kept in development. This split is entirely separate
from the frozen research experiment. File fingerprints are checked before and
during processing; verification requires unchanged code, OCR weights, and audit.

## Comparison protocol and results

Both engines read the same original RGB images, without engine-specific cropping
or tuning. Shared geometric line ordering and `gcash_candidates_v1` parsing follow
OCR. RapidOCR uses ONNX Runtime CPU with intra/inter threads 2/1. Tesseract uses
English fast weights, LSTM-only, sparse-text segmentation (PSM 11), with
`OMP_THREAD_LIMIT=2`. All model files were provisioned before processing; the
benchmark blocks Python socket connections during receipt processing. Images and
raw OCR stay in ignored local storage.

All 144 eligible files completed with both engines and no processing exceptions.
**Only 21 screenshots have visually checked field labels**: 13 purposive development
examples and 8 verification examples selected by filename before inspecting their
images or per-image OCR results. One representative per duplicate group was scored.
These small samples do not estimate general accuracy across GCash layouts/devices.
The labels were read visually by the assistant, not independently double-annotated
by human researchers. Two development transcription errors were corrected after
reinspection; the original annotations and scores are retained with the corrected
version. No parser changes were made from the verification results.

Required-field match means workflow, category, principal, date, and time all match
the visual annotation. Reference is scored separately. A correctly unavailable
field is an abstention, not permission to predict with incomplete inputs.

| Measurement | RapidOCR | Tesseract |
| --- | ---: | ---: |
| Development required-field matches | 12/13 | 6/13 |
| Development exact references | 13/13 | 10/13 |
| Verification required-field matches | 5/8 | 3/8 |
| Verification exact references | 8/8 | 6/8 |
| Development median / p95 seconds per image (110 files) | 1.47 / 1.85 | 0.27 / 0.43 |
| Verification median / p95 seconds per image (34 files) | 1.39 / 2.03 | 0.30 / 0.44 |
| Engine initialization seconds, development / verification | 4.04 / 1.69 | 0.09 / 0.04 |

Per-image timing includes decoding, OCR, and line ordering, excludes parser time
and engine initialization, and is one sequential run per engine, not a throughput
or controlled hardware study. Counts include duplicate files for timing only.

| Folder / checked split | RapidOCR required matches | Tesseract required matches |
| --- | ---: | ---: |
| TRANSFER development | 6/6 | 0/6 |
| PAYMENT development | 5/5 | 4/5 |
| DEBIT development | 1/2 | 2/2 |
| TRANSFER verification | 3/3 | 0/3 |
| PAYMENT verification | 2/3 | 2/3 |
| DEBIT verification | 0/2 | 1/2 |

Across 21 checked images, RapidOCR had 3 missing dates and 3 missing times,
one missing category, and one unexpected category; no wrong non-null numeric/date/
reference values were observed. Tesseract had 9 missing workflows, 10 missing
categories, one missing amount, one unexpected category, and 5 wrong references.
Both correctly left one ambiguous workflow unresolved. A missing/unexpected
category is included in these totals even when the text recognition itself worked:
these are end-to-end OCR-plus-parser results, not pure OCR character accuracy.

RapidOCR's advantage on the checked Express Send headings and references outweighs
Tesseract's lower latency for the first integration. Tesseract read the checked bank
dates better. Retain it as a benchmark comparator; an automatic fallback or hybrid
has not been tested or selected. Neither baseline is ready for automatic acceptance.

## Layout evidence and implementation consequences

- Express Send: checked examples use a heading, separate Amount and Total Amount
  Sent, and a reference/date footer. This is the first integration target.
- Pay Online: checked merchant receipts include linked-wallet and payment-confirmed
  variants, date/reference blocks, advertising and notification overlays. Wallet
  funding and merchant role still require confirmation.
- Bank Transfer: checked layouts separate Transfer Amount, fee, total, invoice,
  and reference. Three of four checked bank-screen dates were missing from the
  RapidOCR/parser result. Users must be able to complete missing date/time.
- `DEBIT/debit_0002.jpg` names **Maya Wallet** as the destination inside the Bank
  Transfer screen. The baseline suggests DEBIT from the heading, which is not a
  valid confirmed mapping under the bank-account-only scope. Flag destination-role
  conflict before prediction; do not silently treat every Bank Transfer destination
  as a bank account or extend Express Send support to other wallets.
- `PAYMENT/payment_0006.jpg` shows a merchant payment under an embedded merchant-app
  header. PAYMENT is visually evident, but the exact selected QR/Pay Online workflow
  cannot be established from the visible heading; leave it unresolved for review.
- No merchant QR layout has been visually verified in this benchmark. The parser
  has candidate cues only; those are not evidence of supported QR receipt handling.
- CASH_IN/CASH_OUT have no samples. Keep both pending and collect 3–5 representative
  app screenshots when available; do not fabricate screenshots or reuse other types.

Follow-on update: the [strict confirmed-input contract and versioned mapping](RECEIPT_MAPPING.md)
with shared-scaler and paired-prediction parity are now implemented. Next are image
jobs, correction/confirmation UI, paired
prediction/explanation, ZIP export and Clear/restart cleanup. Preserve source
workflow, actual account roles, extraction and correction provenance. Do not route
candidate parser dictionaries directly into the trained models. Release each
workflow only after its full upload-to-cleanup journey passes checks. Further parser
tuning must use development data; these verification examples are now seen, so use
new untouched examples for an independent check of future improvements.

## Reproduce locally

Source: `backend/scripts/benchmark_receipts.py`, `receipts/audit.py`, `receipts/ocr.py`,
and `receipts/gcash.py`. The main ML environment and its lock are unchanged. Use the
separate Windows x64 CPython 3.12 snapshot
[`requirements-ocr-benchmark.lock`](../backend/requirements-ocr-benchmark.lock).
It pins installed versions; unlike the main lock, it does not hash every package.
The Tesseract wheel URL is hash-pinned and bundles Tesseract 5.5.2.

From the project root (setup requires network access, receipt processing does not):

```powershell
py -3.12 -m venv backend/runtime/receipt_tools/venv
$receiptPython = '.\backend\runtime\receipt_tools\venv\Scripts\python.exe'
& $receiptPython -m pip install -r backend/requirements-ocr-benchmark.lock
New-Item -ItemType Directory -Force backend/runtime/receipt_tools/tessdata | Out-Null
Invoke-WebRequest -Uri 'https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/4.1.0/eng.traineddata' -OutFile backend/runtime/receipt_tools/tessdata/eng.traineddata
Get-FileHash backend/runtime/receipt_tools/tessdata/eng.traineddata -Algorithm SHA256
& $receiptPython -m pip check
```

Verify the English weights SHA256 is
`7d4322bd2a7749724879683fc3912cb542f19906c83bcc1a52132556427170b2`.
RapidOCR 3.9.2 includes the local model files used here:

| Weight | SHA256 |
| --- | --- |
| PP-OCRv6_det_small.onnx | `090f04abcd9d9a7498bc4ebf677e4cb9bdce1fe4197ddb7e529f1ef44e1ff94f` |
| ch_ppocr_mobile_v2.0_cls_mobile.onnx | `e47acedf663230f8863ff1ab0e64dd2d82b838fceb5957146dab185a89d6215c` |
| PP-OCRv6_rec_small.onnx | `6f327246b50388f3c176ae304bd95767ea6dc0c9ae92153ef8cbe210b3c14884` |

Run with fresh output names; the script refuses to overwrite reports:

```powershell
& $receiptPython backend/scripts/benchmark_receipts.py audit --samples backend/data/raw/receipt_samples --output backend/runtime/receipt_checks/audit_new.json
& $receiptPython backend/scripts/benchmark_receipts.py run --samples backend/data/raw/receipt_samples --audit backend/runtime/receipt_checks/audit_new.json --output backend/runtime/receipt_checks/development_new --tessdata backend/runtime/receipt_tools/tessdata --split development
& $receiptPython backend/scripts/benchmark_receipts.py run --samples backend/data/raw/receipt_samples --audit backend/runtime/receipt_checks/audit_new.json --output backend/runtime/receipt_checks/verification_new --tessdata backend/runtime/receipt_tools/tessdata --split verification --freeze-from backend/runtime/receipt_checks/development_new
```

Current private evidence lives under ignored `backend/runtime/receipt_checks/`:
`audit_20261001.json`, `development_01/`, `verification_01/`,
`annotations_development_v2.json`, `annotations_verification.json`,
`development_score_02.json`, `verification_score_01.json`, and
`engine_decision_before_verification.json`. Raw OCR and annotations can contain
personal receipt data: do not commit them. Synthetic tests are in
`backend/tests/test_receipts.py` and do not load private images or OCR weights.

Verification for this change: 12 synthetic receipt tests passed, both Python
environments passed `pip check`, `git diff --check` passed, and all 146 original
file fingerprints still match the audit. At the baseline checkpoint, development and verification code
fingerprints matched the recorded runs. Subsequent confirmed-mapping work adds and
changes receipt modules, so the current whole-package fingerprint differs. The OCR
adapters/parser and retained baseline results are unchanged; the old development
run cannot serve as a freeze source for new-code verification runs. Main model dependencies, model artifacts
and research data were not changed. The complete backend suite was not rerun at
the isolated benchmark checkpoint; the subsequent [mapping milestone](RECEIPT_MAPPING.md)
passed the full suite (293 tests).

To score a completed run, supply independently visually read annotations with
`annotation_method: "visual_inspection_of_originals"` and `rows` containing `path`,
`sha256`, `split`, and `fields` (`workflow`, `category`, `amount`, `date`, `time`,
`reference`; use strings, or null when visibly unavailable). Amounts use two decimal
places, dates ISO format, time HH:MM in Manila, and reference stays a string.
Use `score --run <run-directory> --annotations <private-json> --output <new-json>`.
Do not generate annotation values by copying OCR output.

Setup references: [RapidOCR installation](https://rapidai.github.io/RapidOCRDocs/main/en/install_usage/rapidocr/install/),
[tesserocr installation and Windows wheels](https://pypi.org/project/tesserocr/),
[Tesseract fast English data](https://github.com/tesseract-ocr/tessdata_fast/tree/4.1.0).
