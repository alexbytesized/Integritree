# Connected receipt application

Implemented 2026-10-01. The receipt flow now connects an uploaded image to local
RapidOCR, editable confirmation, both saved models, SHAP contribution bar charts, ZIP download
and temporary cleanup. Ground Truth and Outcome are absent from receipt results;
researcher records retain them. Both screens share the threshold marker, score
bands, model tabs and SHAP modal. The current saved common cutoff is 43%.

## Available workflows and limits

| Recognized workflow | Initial category |
| --- | --- |
| Express Send | TRANSFER |
| Pay Online | PAYMENT |
| Bank Transfer Complete | DEBIT |

The restored seven-row form exposes reference (Transaction ID), PHP principal amount,
category, Manila date/time, sender type and recipient type. All fields are editable;
roles offer only Client/Merchant. All five model categories can be selected, with
editable role defaults documented in [mapping v2](RECEIPT_MAPPING.md). Proceed confirms
valid fields without review/funding checkboxes. Missing extracted fields stay blank.
Drafts are stored by receipt/revision in sessionStorage and survive refresh.

Receipt extraction and prediction use the shared researcher loading screen. The
confirmation form displays and submits time as `HH:mm`; the amount information
popover explains that the PHP principal excludes fees. Clear Receipt sits beside
Proceed. The background uses the home particle configuration and resizes its canvas
when asynchronous content changes the page height. Results share the researcher
heading and show the optional Transaction ID (`Not provided` when absent). All four
tabs end with Download Results, Edit Details and Clear Results, in that order.

The server still checks multiple OCR layout anchors before field confirmation.
Ambiguous, failed/pending, unsupported, and known wallet/mixed-bank destinations are
rejected. Cash In/Out and merchant QR screenshot recognition remain outside this update.
PaySim fraud labels occur only in TRANSFER and CASH_OUT; these receipt predictions
are an experimental demonstration, not validated GCash fraud detection.

See [mapping](RECEIPT_MAPPING.md), [scope and samples](RECEIPT_WORKFLOW.md) and
[OCR baseline](RECEIPT_OCR_BENCHMARK.md). Missing fields can be completed only after
a supported image is recognized. Confirm PHP principal excluding fees, Manila
date/time and account roles. Reference is optional and preserved
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
3. Review or edit the seven fields, then choose Proceed to confirm.
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
| `GET /{id}/waterfall/{rf\|rf_smote}?revision=N` | Current contribution SVG; optional `layout=modal` omits standalone summary cards (legacy route name). |
| `GET /{id}/download?revision=N` | Current completed result ZIP. |
| `DELETE /{id}` | 204 retires a receipt, including busy jobs; active writers finish before physical deletion. |
| `DELETE /sessions/current` | Idempotent 204 invalidates the session and retires all owned receipt data. |
| `WS /sessions/presence` | Allowed-origin handshake; first JSON message `{token}`, then `{status: "connected"}` acknowledgment. Token is never in the URL. |

`fields` follows [ConfirmedReceiptFields](RECEIPT_MAPPING.md): strict workflow,
category, decimal-string amount, date, time and Client/Merchant roles; optional
reference/names. PHP and Asia/Manila are fixed. Source/layout context is server-owned
and cannot be supplied by a client. Confirmation bodies are limited to 16 KiB.
Errors identify invalid fields without echoing submitted values or exception context.

States: `uploading`, `queued`, `extracting`, `awaiting_confirmation`, `unsupported`,
`failed`, `predicting`, `explaining`, `complete`. Busy jobs cannot be reconfirmed,
but can be cleared. One receipt is retained per session. One worker processes receipt
jobs, with capacity for two queued/running jobs. Session presence is required by the
browser before upload. Missing/expired session returns 410; another
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

The downloaded `integritree_receipt.zip` contains only
`Transaction-Results_YYYY-MM-DD.pdf`. The filename uses the Philippine date (UTC+8),
captured once when each export starts. The PDF has four A4 landscape pages in this
order, regardless of the selected results tab: Benchmark RF Results, Benchmark RF
SHAP Evaluation, RF-SMOTE Results, and RF-SMOTE SHAP Evaluation. Existing results
pages retain the transaction reference, risk gauge and threshold, prediction,
score, interpretation, and SHAP explanation summary.

Each evaluation page repeats its model/transaction header and adds a vector chart
of all eleven features in fixed order, without a chart section title. Red bars
increase risk; green bars decrease it. Values use three significant digits in
percentage points. Exact-zero rows are blank; tiny effects retain their true bar
length and a visible hairline marker. The graph always has nine gridlines and
includes zero and every contribution. It starts at -40 to +40 with 10-point
intervals, shifts when needed (for example -30 to +50), and uses 20- or 25-point
intervals only if an 80-point span is insufficient. Tick-aligned bounds remain
within -100 to +100; the nearest-to-zero center wins, with the lower starting
bound breaking ties.

Reference Score and Output Score cards sit side by side below the graph, followed
by a full-width Top Risk-Increasing Contributor card. Their blue tabs match the
results-page cards. All body values are centered, regular-weight dark text, with
scores shown as percentages to two decimal places and no explanatory definitions.
Missing references show `Not provided`. Pending, failed, or incomplete SHAP data
keeps all four pages, shows an unavailable message instead of an invented graph,
and displays `Unavailable` for missing summary values. Predictions remain available.
PDFs contain no interactive controls. The modal and standalone SVG presentations
are unchanged; the adaptive axis applies only to PDF evaluation pages.

JSON and SVG attachments are not included, and downloads do not render standalone
SVGs. The application retains its confirmed inputs, predictions, explanations,
and metadata, and the authorized SVG endpoints remain available to the UI.
Previously downloaded ZIPs are unchanged.

PDFs use ReportLab and embedded DejaVu Sans fonts from the existing Matplotlib
installation. Install the updated backend dependencies with
`.venv/Scripts/python.exe -m pip install --require-hashes -r requirements-dev.lock`
from `backend/`, then restart the backend. No LibreOffice or browser converter is
needed for receipt PDFs. `pypdf` is a development dependency for export checks.

Export captures the completed revision and a deep copy of the saved result under the session lock,
then renders entirely in memory outside that lock. It does not repeat prediction
or SHAP calculations. Ownership and revision are checked again before returning
the ZIP, so a cleared, expired, or edited receipt cannot deliver a stale export.
PDF failure returns a retryable download error instead of a partial ZIP. No PDF
or ZIP is saved in the session directory. Researcher exports are unchanged.

Private application files live under ignored `backend/runtime/receipt_sessions/`.
Return from confirmation to Upload, browser navigation out of the flow, and Clear
invalidate access immediately. Active upload/OCR/prediction/SHAP writers retain their
files until they finish; their results cannot be revived. Failed file deletion is
retried by the cleanup worker and explicit cleanup offers a retry action.

A live authenticated WebSocket keeps a session alive. The last disconnect starts a
30-second grace period; reconnecting on refresh cancels it. Switching tabs does not
end presence. Unconnected new sessions also expire after 30 seconds. Transport
ping/pong detects broken connections (20-second interval and timeout by default).
Network loss, browser suspension or sleep lasting beyond detection plus grace can
expire a receipt; reconnecting after expiry returns to Upload. A duplicated tab that
shares the same session keeps it alive until its last connection closes.

Use the installed `websockets` transport (in the dependency lock). Vite proxies both
HTTP and WebSocket upgrades. Production origins must be listed in `cors_origins`;
the WebSocket route validates them separately from HTTP CORS. Restart the backend
and load the updated frontend together; temporary v1 sessions are discarded.
Graceful shutdown removes this process's session directory; startup removes abandoned
session directories under the existing runtime lock. Cleanup never touches samples,
datasets, models, research reports or already downloaded ZIPs.

## Verification and remaining acceptance work

Mapping-v2 verification (2026-10-01): the full backend suite passed 329 tests;
receipt/research display browser regressions, two real HTTP/WebSocket lifecycle
checks, a cleanup-failure/late-confirmation race check, and the opt-in local OCR /
saved-model / SHAP / ZIP flow passed. Frontend lint/build and `pip check` passed.
Desktop/mobile screenshots verified the restored seven-row form and local particle
background. Private OCR screenshots/downloads remain under ignored runtime outputs.

UI refinements (2026-10-01): 19 receipt/researcher browser checks passed, covering
shared loading, amount help, minute-only time, particle sizing, and all four results
tabs and footer actions. The opt-in live OCR check was skipped in this frontend run.
Frontend lint and production build passed.


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


### Repeat the real browser lifecycle checks

Run the synthetic backend on port 8001:
`backend/.venv/Scripts/python -m uvicorn receipt_browser_app:create_browser_app --app-dir backend/tests --factory --host 127.0.0.1 --port 8001`.
In a second terminal set `INTEGRITREE_API_TARGET=http://127.0.0.1:8001` and run Vite
on `127.0.0.1:5174`. From frontend, set `RECEIPT_LIFECYCLE_LIVE=1` and
`PLAYWRIGHT_BASE_URL=http://127.0.0.1:5174`, then run
`npx playwright test e2e/receipt-lifecycle.spec.js`.
This fixture uses synthetic OCR and models in a temporary directory; it never reads
private receipt samples or retained research data.
