"""Receipt HTTP lifecycle with synthetic images/outputs; no real dataset or models."""

import io
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from pypdf import PdfReader

# Re-export synthetic fixtures so pytest can resolve them in this module.
from test_research_api import bundle as bundle
from test_research_api import experiment as experiment

from integritree.api.main import create_app
from integritree.ml.features import FEATURE_COLUMNS
from integritree.receipts.gcash import parse_candidates
from integritree.receipts.layouts import identify_layout
from integritree.services import receipt_pdf, receipts

PREFIX = "/api/v1/receipts"
TEXTS = [
    "Express Send",
    "Sent via GCash",
    "Amount 100.00",
    "Total Amount Sent 100.00",
    "Ref No. 0000123",
    "Oct 5, 2026 1:00 PM",
]


def extraction(texts=TEXTS):
    lines = [
        {"text": text, "confidence": 0.9, "box": [0, i * 30, 100, i * 30 + 20]}
        for i, text in enumerate(texts)
    ]
    return {
        "lines": lines,
        "parsed": parse_candidates(lines),
        "extractor_version": "synthetic_ocr_v1",
    }


def png():
    out = io.BytesIO()
    Image.new("RGB", (16, 16), "white").save(out, format="PNG")
    return out.getvalue()


def wait(client, headers, identifier):
    end = time.monotonic() + 15
    while time.monotonic() < end:
        response = client.get(f"{PREFIX}/{identifier}", headers=headers)
        value = response.json()
        if not value["busy"]:
            return value
        time.sleep(0.02)
    pytest.fail("Receipt job did not finish")


class Explainer:
    identity = {"kind": "synthetic"}

    def explain(self, features, ids, analysis_id):
        return [
            {
                "model": model,
                "base_value": 0.1,
                "output_value": 0.2,
                "transaction_id": ids[0],
                "analysis_id": analysis_id,
                "predicted_label": 0,
                "threshold": 0.43,
                "additivity_error": 0,
                "features": [
                    {
                        "feature": name,
                        "readable_value": name,
                        "contribution": 0.1 if i == 0 else 0,
                    }
                    for i, name in enumerate(FEATURE_COLUMNS)
                ],
            }
            for model in ("rf", "rf_smote")
        ]


@pytest.fixture
def client(settings, bundle, experiment):
    app = create_app(settings)
    with TestClient(app) as client:
        app.state.research.bundle = bundle
        app.state.receipts.experiment = experiment
        app.state.receipts.extractor = lambda _: extraction()
        app.state.receipts.explanation_factory = Explainer
        yield client


def session(client):
    return {"X-Receipt-Session": client.post(PREFIX + "/sessions").json()["token"]}


def upload(client, headers, body=None):
    response = client.post(
        PREFIX + "?filename=synthetic.png",
        headers=headers | {"Content-Type": "image/png"},
        content=png() if body is None else body,
    )
    assert response.status_code == 202, response.text
    identifier = response.json()["id"]
    return identifier, wait(client, headers, identifier)


def confirmation(revision=0, **changes):
    return {
        "confirmed": True,
        "expected_revision": revision,
        "fields": {
            "workflow": "express_send",
            "category": "TRANSFER",
            "amount": "100.00",
            "date": "2026-10-05",
            "time": "13:00",
            "reference": None,
            "origin_role": "client",
            "destination_role": "client",
        }
        | changes,
    }


def completed_receipt(client):
    owner = session(client)
    identifier, _ = upload(client, owner)
    path = f"{PREFIX}/{identifier}"
    client.post(path + "/confirm", headers=owner, json=confirmation())
    assert wait(client, owner, identifier)["status"] == "complete"
    return owner, identifier, path


def test_pdf_export_failure_is_retryable_without_recomputing(client, monkeypatch):
    owner, _, path = completed_receipt(client)
    before = client.get(path, headers=owner).json()
    original = receipt_pdf.generate_pdf

    def broken(_):
        raise RuntimeError("private PDF failure")

    monkeypatch.setattr(receipt_pdf, "generate_pdf", broken)
    response = client.get(path + "/download?revision=1", headers=owner)
    assert response.status_code == 503
    assert "Try Download Results again" in response.json()["detail"]
    assert "private PDF failure" not in response.text
    assert client.get(path, headers=owner).json() == before
    monkeypatch.setattr(receipt_pdf, "generate_pdf", original)
    service = client.app.state.receipts
    monkeypatch.setattr(service, "_predict", broken)
    monkeypatch.setattr(service, "_explain", broken)
    monkeypatch.setattr(service, "_contribution_chart", broken)
    response = client.get(path + "/download?revision=1", headers=owner)
    assert response.status_code == 200
    assert len(zipfile.ZipFile(io.BytesIO(response.content)).namelist()) == 1
    assert client.get(path, headers=owner).json() == before
    assert not list(service.folder.rglob("*.pdf"))
    assert not list(service.folder.rglob("*.zip"))


@pytest.mark.parametrize("explanation_status", ["computed", "pending", "failed"])
def test_pdf_only_download_uses_snapshot_without_rendering_svg(
    client, monkeypatch, explanation_status
):
    owner, identifier, path = completed_receipt(client)
    service = client.app.state.receipts
    with service.lock:
        if explanation_status != "computed":
            service.jobs[identifier]["result"]["explanation"] = {
                "status": explanation_status
            }
    before = client.get(path, headers=owner).json()

    def forbidden(*args, **kwargs):
        raise AssertionError("Downloads must use the saved snapshot only")

    monkeypatch.setattr(service, "_contribution_chart", forbidden)
    monkeypatch.setattr(service, "bundle_provider", forbidden)
    monkeypatch.setattr(service, "_predict", forbidden)
    monkeypatch.setattr(service, "_explain", forbidden)
    response = client.get(path + "/download?revision=1", headers=owner)
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    assert (
        'filename="integritree_receipt.zip"' in response.headers["content-disposition"]
    )
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert len(archive.namelist()) == 1
        name = archive.namelist()[0]
        assert name.startswith("Transaction-Results_") and name.endswith(".pdf")
        reader = PdfReader(io.BytesIO(archive.read(name)))
        assert len(reader.pages) == 4
        if explanation_status != "computed":
            for index in (1, 3):
                assert "unavailable" in reader.pages[index].extract_text().lower()
    assert client.get(path, headers=owner).json() == before


def test_chart_layouts_preserve_receipt_revision_and_ownership(client):
    owner, _, path = completed_receipt(client)
    endpoint = path + "/waterfall/rf?revision=1"
    standalone = client.get(endpoint, headers=owner)
    modal = client.get(endpoint + "&layout=modal", headers=owner)
    assert standalone.status_code == modal.status_code == 200
    assert "Reference" in standalone.text and "Reference" not in modal.text
    assert "Hour of the Day" not in modal.text and "#a00000" in modal.text
    assert client.get(endpoint + "&layout=bad", headers=owner).status_code == 422
    assert (
        client.get(endpoint + "&layout=modal", headers=session(client)).status_code
        == 404
    )
    assert (
        client.get(
            path + "/waterfall/rf?revision=2&layout=modal", headers=owner
        ).status_code
        == 409
    )


def test_concurrent_pdf_exports_keep_session_snapshots_separate(client, monkeypatch):
    receipts_to_export = []
    for reference in ("FIRST-OWNER", "SECOND-OWNER"):
        owner = session(client)
        identifier, _ = upload(client, owner)
        path = f"{PREFIX}/{identifier}"
        client.post(
            path + "/confirm", headers=owner, json=confirmation(reference=reference)
        )
        assert wait(client, owner, identifier)["status"] == "complete"
        receipts_to_export.append((owner, path, reference))
    barrier = threading.Barrier(2)
    original = receipt_pdf.generate_pdf

    def simultaneous(result):
        barrier.wait(timeout=10)
        return original(result)

    monkeypatch.setattr(receipt_pdf, "generate_pdf", simultaneous)
    with ThreadPoolExecutor() as pool:
        futures = [
            pool.submit(client.get, path + "/download?revision=1", headers=owner)
            for owner, path, _ in receipts_to_export
        ]
        responses = [future.result(timeout=15) for future in futures]
    for response, (_, _, reference) in zip(responses, receipts_to_export):
        assert response.status_code == 200
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            assert len(archive.namelist()) == 1
            name = next(n for n in archive.namelist() if n.endswith(".pdf"))
            reader = PdfReader(io.BytesIO(archive.read(name)))
            assert len(reader.pages) == 4
            for page in reader.pages:
                text = page.extract_text()
                assert reference in text
                assert (
                    "SECOND-OWNER" if reference == "FIRST-OWNER" else "FIRST-OWNER"
                ) not in text


def test_repeated_exports_capture_philippine_date_once(client, monkeypatch):
    owner, _, path = completed_receipt(client)

    class Clock:
        instant = datetime(2026, 10, 5, 15, 59, 59, tzinfo=UTC)
        calls = 0

        @classmethod
        def now(cls, zone):
            cls.calls += 1
            return cls.instant.astimezone(zone)

    monkeypatch.setattr(receipts, "datetime", Clock)
    original = receipt_pdf.generate_pdf

    def crossing_midnight(result):
        Clock.instant = datetime(2026, 10, 5, 16, tzinfo=UTC)
        return original(result)

    monkeypatch.setattr(receipt_pdf, "generate_pdf", crossing_midnight)
    for day in ("05", "06"):
        response = client.get(path + "/download?revision=1", headers=owner)
        assert response.status_code == 200
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            assert archive.namelist() == [f"Transaction-Results_2026-10-{day}.pdf"]
    assert Clock.calls == 2


@pytest.mark.parametrize("change", ["clear", "edit", "expire"])
def test_pdf_export_revalidates_after_render_and_does_not_block_clear(
    client, monkeypatch, change
):
    owner, identifier, path = completed_receipt(client)
    service = client.app.state.receipts
    folder = service.jobs[identifier]["folder"]
    entered, release = threading.Event(), threading.Event()
    original = receipt_pdf.generate_pdf

    def delayed(result):
        entered.set()
        assert release.wait(10)
        return original(result)

    monkeypatch.setattr(receipt_pdf, "generate_pdf", delayed)
    with ThreadPoolExecutor() as pool:
        exporting = pool.submit(
            client.get, path + "/download?revision=1", headers=owner
        )
        try:
            assert entered.wait(10)
            outsider = session(client)
            assert (
                client.get(path + "/download?revision=1", headers=outsider).status_code
                == 404
            )
            if change == "edit":
                assert (
                    client.post(
                        path + "/confirm",
                        headers=owner,
                        json=confirmation(1, reference="updated"),
                    ).status_code
                    == 202
                )
                assert wait(client, owner, identifier)["revision"] == 2
            elif change == "clear":
                assert client.delete(path, headers=owner).status_code == 204
                assert not folder.exists()
            else:
                service.reap(now=time.monotonic() + 60)
                assert not folder.exists()
        finally:
            release.set()
        response = exporting.result(timeout=10)
    assert response.status_code == {"edit": 409, "clear": 404, "expire": 410}[change]
    assert not response.content.startswith(b"PK")


def test_upload_confirmation_results_export_revision_and_clear(client):
    headers = session(client)
    identifier, status = upload(client, headers)
    path = f"{PREFIX}/{identifier}"
    assert status["status"] == "awaiting_confirmation" and status["revision"] == 0
    assert status["candidates"]["reference"] == "0000123"
    assert "result" not in status
    preview = client.get(path + "/image", headers=headers)
    assert preview.content == png() and preview.headers["Cache-Control"] == "no-store"
    assert (
        client.post(path + "/confirm", headers=headers, json=confirmation()).status_code
        == 202
    )
    status = wait(client, headers, identifier)
    assert status["status"] == "complete", status
    result = status["result"]
    assert result["threshold"] == 0.43 and result["rf"]["predicted_label"] == 1
    assert result["explanation"]["status"] == "computed"
    assert (
        result["derived"]["day_of_week"] == 0
        and result["original"]["reference"] is None
    )
    assert not {"actual_label", "ground_truth", "outcome", "evaluation"}.intersection(
        result
    )
    assert (
        client.get(path + "/waterfall/rf?revision=1", headers=headers).status_code
        == 200
    )
    archive = client.get(path + "/download?revision=1", headers=headers)
    assert archive.status_code == 200
    with zipfile.ZipFile(io.BytesIO(archive.content)) as z:
        pdfs = [name for name in z.namelist() if name.endswith(".pdf")]
        assert len(pdfs) == 1
        assert pdfs[0] == receipt_pdf.pdf_filename(datetime.now(UTC))
        assert z.namelist() == pdfs
        assert len(PdfReader(io.BytesIO(z.read(pdfs[0]))).pages) == 4
    assert client.get(path, headers=headers).json() == status
    for model in ("rf", "rf_smote"):
        chart = client.get(path + f"/waterfall/{model}?revision=1", headers=headers)
        assert chart.status_code == 200
        assert b"Contribution to Fraud Risk Score (Percentage Points)" in chart.content
    assert (
        client.post(path + "/confirm", headers=headers, json=confirmation()).status_code
        == 409
    )
    assert (
        client.post(
            path + "/confirm",
            headers=headers,
            json=confirmation(
                1,
                amount="0",
                reference="0009",
                category="CASH_IN",
                origin_role="merchant",
            ),
        ).status_code
        == 202
    )
    assert client.get(path + "/download?revision=1", headers=headers).status_code == 409
    updated = wait(client, headers, identifier)["result"]
    assert updated["revision"] == 2 and updated["derived"]["is_zero_amount"] == 1
    assert updated["input_sha256"] != result["input_sha256"]
    assert (
        updated["derived"]["type_CASH_IN"] == 1
        and updated["derived"]["is_merchant_origin"] == 1
    )
    assert updated["field_provenance"]["category"]["status"] == "corrected"
    folder = client.app.state.receipts.jobs[identifier]["folder"]
    assert not (folder / "revision_1").exists()
    assert (
        client.get(path + "/waterfall/rf?revision=1", headers=headers).status_code
        == 409
    )
    assert client.delete(path, headers=headers).status_code == 204
    assert not folder.exists()
    assert client.get(path, headers=headers).status_code == 404


def test_session_isolation_expiry_and_untrusted_confirmation(client):
    owner, other = session(client), session(client)
    identifier, _ = upload(client, owner)
    path = f"{PREFIX}/{identifier}"
    for suffix in ("", "/image", "/download?revision=1", "/waterfall/rf?revision=1"):
        assert client.get(path + suffix, headers=other).status_code == 404
        assert client.get(path + suffix).status_code == 410
    assert (
        client.post(path + "/confirm", headers=other, json=confirmation()).status_code
        == 404
    )
    assert client.post(path + "/retry", headers=other).status_code == 404
    assert client.delete(path, headers=other).status_code == 404
    for payload in (
        confirmation() | {"source": {"layout_id": "gcash_pay_online_v1"}},
        confirmation() | {"confirmed": False},
        confirmation(amount="PRIVATE_INVALID_AMOUNT"),
        confirmation(
            workflow="bank_transfer", category="DEBIT", destination_role="bank_account"
        ),
    ):
        response = client.post(path + "/confirm", headers=owner, json=payload)
        assert response.status_code == 422
        assert "PRIVATE_INVALID_AMOUNT" not in response.text
        assert all("input" not in issue for issue in response.json()["issues"])


def test_invalid_image_and_unsupported_layout_cannot_be_overridden(client):
    owner = session(client)
    identifier, job = upload(client, owner, b"not an image")
    assert job["status"] == "unsupported"
    assert job.get("error_code") is None
    assert (
        client.post(
            f"{PREFIX}/{identifier}/confirm", headers=owner, json=confirmation()
        ).status_code
        == 409
    )
    client.delete(f"{PREFIX}/{identifier}", headers=owner)
    client.app.state.receipts.extractor = lambda _: extraction(
        ["PAYMENT", "Amount 100.00"]
    )
    identifier, job = upload(client, owner)
    assert job["status"] == "unsupported"
    assert job["error_code"] == "unsupported_receipt_layout"
    assert job["filename"] == "synthetic.png"
    assert (
        client.post(
            f"{PREFIX}/{identifier}/confirm", headers=owner, json=confirmation()
        ).status_code
        == 409
    )


@pytest.mark.parametrize(
    "texts,message",
    [
        (TEXTS + ["Pending"], "Only completed transaction"),
        (
            ["Bank Transfer Complete", "Transfer Date", "Transfer Amount"],
            "destination bank",
        ),
    ],
)
def test_specific_layout_rejections_keep_their_message(client, texts, message):
    client.app.state.receipts.extractor = lambda _: extraction(texts)
    _, job = upload(client, session(client))
    assert job["status"] == "unsupported"
    assert job.get("error_code") is None
    assert message in job["error"]


def test_size_limits_and_upload_abort_release_slot(client, monkeypatch):
    import integritree.api.routes.receipts as routes

    monkeypatch.setattr(routes, "MAX_BYTES", 100)
    owner = session(client)
    assert (
        client.post(
            PREFIX + "?filename=x.png",
            content=b"a" * 101,
            headers=owner | {"Content-Type": "image/png"},
        ).status_code
        == 413
    )
    assert (
        client.post(
            PREFIX + "?filename=x.png",
            content=iter([b"a" * 60, b"b" * 60]),
            headers=owner | {"Content-Type": "image/png"},
        ).status_code
        == 413
    )
    assert not client.app.state.receipts.jobs
    _, job = upload(client, owner)
    assert job["status"] == "awaiting_confirmation"


def test_busy_cannot_be_confirmed_and_failed_ocr_can_retry(client):
    start, finish = threading.Event(), threading.Event()

    def slow(_):
        start.set()
        assert finish.wait(5)
        raise RuntimeError("private error text")

    client.app.state.receipts.extractor = slow
    owner = session(client)
    response = client.post(
        PREFIX + "?filename=x.png",
        content=png(),
        headers=owner | {"Content-Type": "image/png"},
    )
    identifier = response.json()["id"]
    assert start.wait(5)
    try:
        assert (
            client.post(
                f"{PREFIX}/{identifier}/confirm", headers=owner, json=confirmation()
            ).status_code
            == 409
        )
    finally:
        finish.set()
    failed = wait(client, owner, identifier)
    assert failed["status"] == "failed" and "private error text" not in failed["error"]
    client.app.state.receipts.extractor = lambda _: extraction()
    assert client.post(f"{PREFIX}/{identifier}/retry", headers=owner).status_code == 202
    assert wait(client, owner, identifier)["status"] == "awaiting_confirmation"


def test_shutdown_and_restart_cleanup_preserve_collection(settings, bundle, experiment):
    samples = settings.backend_root / "data/raw/receipt_samples/TRANSFER"
    samples.mkdir(parents=True)
    (samples / "keep.png").write_bytes(png())
    stale = settings.backend_root / "runtime/receipt_sessions/session_interrupted"
    stale.mkdir(parents=True)
    (stale / "private.png").write_bytes(png())
    app = create_app(settings)
    with TestClient(app) as client:
        assert not stale.exists()
        client.app.state.receipts.extractor = lambda _: extraction()
        identifier, _ = upload(client, session(client))
        folder = client.app.state.receipts.jobs[identifier]["folder"]
        assert folder.exists()
    assert not folder.exists() and (samples / "keep.png").exists()


@pytest.mark.parametrize(
    "destination", ["Maya Philippines, Inc. / Maya Wallet", "BPI / VYBE by BPI"]
)
def test_bank_screen_wallet_destinations_are_not_supported(destination):
    lines = extraction(
        [
            "Bank Transfer",
            "Bank Transfer Complete",
            "Bank " + destination,
            "Account No. masked",
            "Transfer Date",
            "Transfer Amount 1.00",
            "Ref No. 0001",
        ]
    )["lines"]
    assert identify_layout(lines)[0] is None


@pytest.mark.parametrize(
    "texts,workflow,category,role",
    [
        (
            [
                "Pay Online",
                "Paid via GCash",
                "Amount 125.50",
                "Date Oct 5, 2026 1:00 PM",
            ],
            "pay_online",
            "PAYMENT",
            "merchant",
        ),
        (
            [
                "Bank Transfer Complete",
                "Bank Example Bank",
                "Account No. masked",
                "Transfer Date",
                "Oct 5, 2026 1:00 PM",
                "Transfer Amount 125.50",
            ],
            "bank_transfer",
            "DEBIT",
            "merchant",
        ),
    ],
)
def test_other_supported_layouts_allow_missing_reference(
    client, texts, workflow, category, role
):
    client.app.state.receipts.extractor = lambda _: extraction(texts)
    owner = session(client)
    identifier, job = upload(client, owner)
    assert job["status"] == "awaiting_confirmation" and job["workflow"] == workflow
    assert job["candidates"]["reference"] is None
    response = client.post(
        f"{PREFIX}/{identifier}/confirm",
        headers=owner,
        json=confirmation(workflow=workflow, category=category, destination_role=role),
    )
    assert response.status_code == 202
    result = wait(client, owner, identifier)["result"]
    assert result["original"]["category"] == category
    assert result["derived"][f"type_{category}"] == 1
    assert result["explanation"]["status"] == "computed"


def test_missing_field_can_be_completed_but_conflicting_layout_cannot():
    texts = [line for line in TEXTS if not line.startswith("Amount")]
    assert identify_layout(extraction(texts)["lines"])[0] == "gcash_express_send_v1"
    for extra in (["Pay Online", "Paid via GCash"], ["Transaction Pending"]):
        assert identify_layout(extraction(TEXTS + extra)["lines"])[0] is None


def test_explanation_failure_keeps_predictions_and_can_retry(client):
    class BrokenExplainer:
        def explain(self, *args):
            raise RuntimeError("private failure")

    service = client.app.state.receipts
    service.explanation_factory = BrokenExplainer
    owner = session(client)
    identifier, _ = upload(client, owner)
    path = f"{PREFIX}/{identifier}"
    client.post(path + "/confirm", headers=owner, json=confirmation())
    job = wait(client, owner, identifier)
    assert job["status"] == "complete" and "rf" in job["result"]
    assert job["result"]["explanation"]["status"] == "failed"
    assert "private failure" not in str(job)
    assert (
        client.get(path + "/waterfall/rf?revision=1", headers=owner).status_code == 409
    )
    response = client.get(path + "/download?revision=1", headers=owner)
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert not any(name.endswith(".svg") for name in archive.namelist())
        assert len(archive.namelist()) == 1
        pdf_name = next(name for name in archive.namelist() if name.endswith(".pdf"))
        reader = PdfReader(io.BytesIO(archive.read(pdf_name)))
        assert len(reader.pages) == 4
        assert "SHAP explanation is unavailable" in reader.pages[0].extract_text()
        assert client.get(path, headers=owner).json() == job
    service.engine = Explainer()
    assert client.post(path + "/retry", headers=owner).status_code == 202
    assert (
        wait(client, owner, identifier)["result"]["explanation"]["status"] == "computed"
    )


def test_revision_cleanup_failure_releases_queue_slot(client, monkeypatch):
    owner = session(client)
    identifier, _ = upload(client, owner)
    path = f"{PREFIX}/{identifier}"
    client.post(path + "/confirm", headers=owner, json=confirmation())
    wait(client, owner, identifier)
    service = client.app.state.receipts
    original = service._remove

    def locked(_):
        raise OSError("private path")

    monkeypatch.setattr(service, "_remove", locked)
    try:
        for _ in range(3):
            assert (
                client.post(
                    path + "/confirm", headers=owner, json=confirmation(1)
                ).status_code
                == 503
            )
        assert service.status(owner["X-Receipt-Session"], identifier)["revision"] == 1
    finally:
        monkeypatch.setattr(service, "_remove", original)


@pytest.mark.parametrize("stage", ["ocr", "predict", "explain"])
def test_session_clear_during_work_prevents_revival_and_releases_capacity(
    client, stage
):
    service = client.app.state.receipts
    owner = session(client)
    token = owner["X-Receipt-Session"]
    entered, finish = threading.Event(), threading.Event()

    def block():
        entered.set()
        assert finish.wait(10)

    if stage == "ocr":
        service.extractor = lambda _: (block(), extraction())[1]
        response = client.post(
            PREFIX + "?filename=x.png",
            content=png(),
            headers=owner | {"Content-Type": "image/png"},
        )
        identifier = response.json()["id"]
    else:
        identifier, _ = upload(client, owner)
        if stage == "predict":
            original = service.bundle_provider
            service.bundle_provider = lambda: (block(), original())[1]
        else:

            class SlowExplainer(Explainer):
                def explain(self, *args):
                    block()
                    return super().explain(*args)

            service.explanation_factory = SlowExplainer
        assert (
            client.post(
                f"{PREFIX}/{identifier}/confirm", headers=owner, json=confirmation()
            ).status_code
            == 202
        )
    assert entered.wait(5)
    job = service.jobs[identifier]
    folder = job["folder"]
    try:
        assert (
            client.delete(PREFIX + "/sessions/current", headers=owner).status_code
            == 204
        )
        assert (
            client.delete(PREFIX + "/sessions/current", headers=owner).status_code
            == 204
        )
        assert token not in service.sessions and identifier not in service.jobs
        assert folder.exists()  # The active writer still owns these files.
        assert client.get(f"{PREFIX}/{identifier}", headers=owner).status_code == 410
    finally:
        finish.set()
    end = time.monotonic() + 10
    while folder.exists() and time.monotonic() < end:
        time.sleep(0.02)
    assert not folder.exists()
    assert job["result"] is None and not job["slot_held"]
    assert service.slots.acquire(blocking=False)
    assert service.slots.acquire(blocking=False)
    assert not service.slots.acquire(blocking=False)
    service.slots.release()
    service.slots.release()
    service.extractor = lambda _: extraction()
    upload(client, session(client))


def test_clear_during_upload_aborts_without_double_releasing_slot(client):
    service = client.app.state.receipts
    owner = session(client)
    job = service.reserve(owner["X-Receipt-Session"], "x.png")
    with (job["folder"] / "image").open("wb") as stream:
        stream.write(png())
        service.delete_session(owner["X-Receipt-Session"])
        assert job["folder"].exists()
    from integritree.services.receipts import ReceiptError

    with pytest.raises(ReceiptError):
        service.submit(job, "a" * 64)
    service.abort(job)
    service.abort(job)
    assert not job["folder"].exists()
    upload(client, session(client))


def test_presence_refresh_multi_connection_and_missed_close_cleanup(client):
    service = client.app.state.receipts
    owner, other = session(client), session(client)
    identifier, _ = upload(client, owner)
    other_id, _ = upload(client, other)
    service.attach(other["X-Receipt-Session"])
    token = owner["X-Receipt-Session"]
    first = service.attach(token)
    second = service.attach(token)
    service.detach(token, first)
    assert token not in service.deadlines
    service.detach(token, second)
    deadline = service.deadlines[token]
    service.reap(deadline - 0.01)
    assert identifier in service.jobs
    refreshed = service.attach(token)
    service.reap(deadline + 1)
    assert identifier in service.jobs
    service.detach(token, refreshed)
    folder = service.jobs[identifier]["folder"]
    service.reap(service.deadlines[token] + 0.01)
    assert identifier not in service.jobs and not folder.exists()
    assert other_id in service.jobs
    orphan = service.session()  # Never establishes presence: no leaked session.
    service.reap(service.deadlines[orphan] + 0.01)
    assert orphan not in service.sessions


def test_websocket_authentication_origin_and_disconnect(client):
    from starlette.websockets import WebSocketDisconnect

    service = client.app.state.receipts
    owner = session(client)
    token = owner["X-Receipt-Session"]
    path = PREFIX + "/sessions/presence"
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect(
            path, headers={"Origin": "https://untrusted.example"}
        ):
            pass
    with client.websocket_connect(
        path, headers={"Origin": "http://localhost:5173"}
    ) as ws:
        ws.send_json({"token": token})
        assert ws.receive_json() == {"status": "connected"}
        assert token not in service.deadlines
    assert token in service.deadlines
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect(
            path, headers={"Origin": "http://localhost:5173"}
        ) as ws:
            ws.send_json({"token": "invalid"})
            ws.receive_json()


def test_session_cleanup_failure_is_inaccessible_and_retryable(client, monkeypatch):
    owner = session(client)
    identifier, _ = upload(client, owner)
    service = client.app.state.receipts
    folder = service.jobs[identifier]["folder"]
    remove = service._remove

    def fail(_):
        raise OSError("private path")

    monkeypatch.setattr(service, "_remove", fail)
    response = client.delete(PREFIX + "/sessions/current", headers=owner)
    assert response.status_code == 503 and "private path" not in response.text
    assert client.get(f"{PREFIX}/{identifier}", headers=owner).status_code == 410
    monkeypatch.setattr(service, "_remove", remove)
    assert client.delete(PREFIX + "/sessions/current", headers=owner).status_code == 204
    assert not folder.exists()
