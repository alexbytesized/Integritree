# Connected receipt application

Implemented 2026-10-01. The receipt flow now connects an uploaded image to local
RapidOCR, editable confirmation, both saved models, SHAP waterfalls, ZIP download
and temporary cleanup. Ground Truth and Outcome are absent from receipt results;
researcher records retain them. Both screens share the threshold marker, score
bands, model tabs and SHAP modal. The current saved common cutoff is 43%.

## Available workflows and limits

| Recognized workflow | Confirmed category | Required destination |
| --- | --- | --- |
| Express Send | TRANSFER | Personal wallet |
| Pay Online | PAYMENT | Merchant |
| Bank Transfer Complete | DEBIT | Bank account |

All require a personal-wallet origin and confirmation that wallet funds paid the
transaction. The server checks multiple OCR layout anchors before allowing field
confirmation. Ambiguous, failed/pending, unsupported, and known wallet/mixed-bank
destinations are rejected. Maya Wallet and BPI/VYBE destination labels cannot be
overridden as bank-account DEBIT. These heuristic gates recognize sampled layouts;
they do not authenticate screenshots or verify the user's account-role statement.

Cash-in, cash-out and merchant QR remain pending samples/layout checks and, for
cash categories, role mapping. The approved five-category target is unchanged.
PaySim fraud labels occur only in TRANSFER and CASH_OUT; these receipt predictions
are an experimental demonstration, not validated GCash fraud detection.

See [mapping](RECEIPT_MAPPING.md), [scope and samples](RECEIPT_WORKFLOW.md) and
[OCR baseline](RECEIPT_OCR_BENCHMARK.md). Missing fields can be completed only after
a supported image is recognized. Confirm PHP principal excluding fees, Manila
date/time, account roles and wallet funding. Reference is optional and preserved
as text. No fake PaySim step/account ID or user-supplied feature matrix is used.

## Run locally

Use the existing backend and frontend commands in [backend setup](../backend/README.md).
Install the isolated OCR environment using the [benchmark setup](RECEIPT_OCR_BENCHMARK.md)
before receipt extraction. Keep those packages out of the frozen ML environment.
`INTEGRITREE_RECEIPT_OCR_PYTHON` defaults to
`backend/runtime/receipt_tools/venv/Scripts/python.exe`; relative settings resolve
against the backend root. RapidOCR runs in a child process with network connections
disabled and a 120-second extraction timeout. Its models must already be installed.

Restart an existing backend process to load the routes. Run one backend worker;
the runtime lock prevents two processes from sharing temporary storage. Open
`http://127.0.0.1:5173/upload` with the frontend and backend running. Vite proxies
`/api` to port 8000. The health endpoint does not verify OCR/model readiness.

1. Upload one PNG/JPEG, at most 10 MiB and 20 million decoded pixels.
2. Review extracted fields and complete/correct them against the screenshot.
3. Confirm the account roles, wallet funding and transaction details.
4. View paired results and SHAP; download the ZIP before Clear or backend shutdown.

The models and saved preprocessing load lazily; no retraining occurs. SHAP uses
the existing training-background policy. Receipt hour/weekday labels say Manila,
while the model's numerical contributions remain unchanged.

## HTTP contract

All routes are under `/api/v1/receipts`. Create a session with
`POST /sessions` (201), then send its opaque `token` in `X-Receipt-Session`.
The browser retains the token and active receipt ID in sessionStorage. Every image,
result, chart and download checks ownership. Responses use `Cache-Control: no-store`.

| Method and relative path | Contract |
| --- | --- |
| `POST ?filename=receipt.png` | Raw PNG/JPEG body, corresponding Content-Type; 202 with `id` and status. Not multipart. Actual streamed bytes are bounded. |
| `GET /{id}` | Job status, `busy`, revision, error, and available workflow/candidates/confirmed fields/result. |
| `GET /{id}/image` | Authenticated validated source-image preview. |
| `POST /{id}/confirm` | JSON `{fields, confirmed: true, expected_revision}`; 202 starts paired prediction and SHAP. Initial revision is 0. |
| `POST /{id}/retry` | 202 retries failed OCR, prediction or explanation work. |
| `GET /{id}/waterfall/{rf\|rf_smote}?revision=N` | Current computed SVG only. |
| `GET /{id}/download?revision=N` | Current completed result ZIP. |
| `DELETE /{id}` | 204 clears an inactive receipt, private files and derived results. |

`fields` follows [ConfirmedReceiptFields](RECEIPT_MAPPING.md): strict workflow,
category, decimal-string amount, date, time, roles and wallet funding; optional
reference/names. PHP and Asia/Manila are fixed. Source/layout context is server-owned
and cannot be supplied by a client. Confirmation bodies are limited to 16 KiB.
Errors identify invalid fields without echoing submitted values or exception context.

States: `uploading`, `queued`, `extracting`, `awaiting_confirmation`, `unsupported`,
`failed`, `predicting`, `explaining`, `complete`. `busy` controls whether Clear or
confirmation is allowed. One receipt is retained per session. One worker processes
receipt jobs, with capacity for two queued/running jobs. Closing a tab does not
cancel work or guarantee deletion. Missing/expired session returns 410; another
session's receipt returns 404; incompatible state/stale revision returns 409;
oversized upload returns 413; invalid media type returns 415; confirmation validation
returns 422; full queue returns 429. Image/layout rejection can be asynchronous.

Every confirmation advances the revision, clears previous derived files/results
and rechecks the stored image fingerprint. Stale revision charts/downloads return
409. Results include `original`, eleven unscaled `derived` fields, `model_inputs`,
mapping/input/model provenance, `threshold`, and explicit `rf`/`rf_smote` scores and
predicted labels. They contain no actual label, correctness outcome or evaluation.
Explanation status is `pending`, `computed` or `failed`. A failed explanation keeps
the predictions available and allows retry; it does not become a fabricated chart.

## Download and retention

The ZIP contains `confirmed_inputs.json`, `results.json`, `explanations.json` and
`metadata.json`. Computed explanations add `rf_waterfall.svg` and
`rf_smote_waterfall.svg`. Metadata identifies the mapping, revision, input fingerprint,
model run, preprocessing, threshold and explanation state. An unavailable explanation
is explicit in the JSON. The ZIP excludes the original image and evaluation metrics.

Private application files live under ignored `backend/runtime/receipt_sessions/`.
Clear deletes the receipt directory; graceful shutdown removes this process's
session directory; startup removes abandoned session directories after acquiring
the runtime lock. Cleanup never touches retained development screenshots, datasets,
models or research reports. Downloaded ZIPs remain wherever the user saves them.

## Verification and remaining acceptance work

Synthetic backend checks cover all three mappings/layouts, ownership, limits,
confirmation, revision invalidation, explanation recovery, ZIP contents and cleanup.
Browser checks cover editable completion, all receipt model tabs without Ground
Truth/Outcome, threshold markers, SHAP, refresh, revision, download, Clear and
unsupported layouts; researcher display regression checks retain labeled behavior.
The full backend suite passed 301 tests before the final cleanup-recovery guard;
the receipt API suite then passed all 13 tests, including five added cases for
missing references, the other layouts, explanation recovery and cleanup failure.
Eight browser regression checks and the separate real-image smoke test passed;
frontend lint and production build passed. Dependency deprecation warnings remain.

A local Express Send screenshot passed real RapidOCR, saved paired inference,
SHAP, download and Clear on an isolated server. This is an integration smoke test
using an existing development sample, not a new accuracy estimate. Private outputs
remain ignored. Retained models and research artifacts were unchanged; no official
held-out evaluation was run. Additional unseen Pay Online and bank-account samples
are still needed for broader acceptance, followed by cash/QR collection and mapping.

Frontend checks: `npm.cmd run lint`, `npm.cmd run build`, and
`npx.cmd playwright test e2e/receipts.spec.js e2e/research-display.spec.js`.
The real-image browser smoke test is opt-in: set `RECEIPT_SAMPLE_PATH` to a consented
local supported screenshot and `RECEIPT_API_BASE_URL` to the running backend, then
run Playwright with `-g 'live local'`. It uploads, confirms and clears that temporary
copy; screenshots/ZIPs in the ignored test-output directory may contain private data.
Use separate `--output` directories for simultaneous Playwright runs.
