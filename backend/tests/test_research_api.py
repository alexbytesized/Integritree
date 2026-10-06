"""Synthetic integration and numerical parity; never opens held-out data."""

import csv
from datetime import datetime, timezone
import hashlib
import io
import sqlite3
import time
from types import SimpleNamespace
import zipfile
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from integritree.api.main import create_app
from integritree.config import load_experiment
from integritree.ml.evaluation import evaluate_pair
from integritree.ml.features import SOURCE_COLUMNS, FEATURE_COLUMNS
from integritree.ml.inference import predict_records
from integritree.ml.preprocessing import FittedPreprocessor
from integritree.services.research import ResearchService, ResearchError, safe_csv
from integritree.services.research_metrics import evaluate_database
from integritree.settings import BACKEND_ROOT

PREFIX = "/api/v1/research"
HEADER = "step,type,amount,nameOrig,nameDest,isFraud\n"
CSV = HEADER + "1,TRANSFER,100,C_A,C_B,1\n2,PAYMENT,0,C_B,M_C,0\n"


def wait_for(fn, predicate, seconds=10):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        value = fn()
        if predicate(value):
            return value
        time.sleep(0.02)
    pytest.fail(f"Job did not finish: {value}")


class Model:
    classes_ = np.array([0, 1])

    def __init__(self, alternate=False):
        self.alternate = alternate

    def predict_proba(self, matrix):
        scores = np.where(
            matrix[:, FEATURE_COLUMNS.index("type_TRANSFER")] == 1, 0.43, 0.1
        )
        if self.alternate:
            scores = np.where(scores == 0.1, 0.7, 0.43)
        return np.column_stack([1 - scores, scores])


@pytest.fixture
def experiment():
    return load_experiment(BACKEND_ROOT / "configs/experiment.yaml")


@pytest.fixture
def bundle(experiment):
    config = experiment.model_copy(deep=True)
    config.scoring.threshold = 0.43
    preprocessor = FittedPreprocessor.fit(pd.read_csv(io.StringIO(CSV))[SOURCE_COLUMNS])
    return SimpleNamespace(
        models={"rf": Model(), "rf_smote": Model(True)},
        preprocessor=preprocessor,
        config=config,
        metadata={"run_id": "synthetic-frozen", "stage": "validation_selected"},
    )


@pytest.fixture
def client(settings, bundle, experiment, monkeypatch):
    # Unit/integration tests exercise template filling without an office runtime.
    # Real conversion and PDF layout are checked separately with synthetic data.
    monkeypatch.setattr(
        "integritree.services.experiment_paper.convert_pdf",
        lambda *args: b"%PDF-1.4\nsynthetic converter test double\n%%EOF",
    )
    app = create_app(settings)
    with TestClient(app) as client:
        app.state.research.bundle = bundle
        app.state.research.experiment = experiment
        yield client


def session(client):
    return {"X-Research-Session": client.post(PREFIX + "/sessions").json()["token"]}


def upload(client, headers, content=CSV, name="demo.csv"):
    result = client.post(
        PREFIX + "/analyses",
        params={"filename": name},
        content=content,
        headers=headers | {"Content-Type": "text/csv"},
    )
    assert result.status_code == 202, result.text
    identifier = result.json()["id"]
    job = wait_for(
        lambda: client.get(PREFIX + f"/analyses/{identifier}", headers=headers).json(),
        lambda value: value["status"] in ("complete", "failed"),
    )
    return identifier, job


def test_complete_flow_isolation_and_cleanup(client, bundle, monkeypatch):
    def no_fit(*args, **kwargs):
        raise AssertionError("Inference must not fit")

    monkeypatch.setattr(FittedPreprocessor, "fit", no_fit)
    monkeypatch.setattr(FittedPreprocessor, "fit_batches", no_fit)
    owner, other = session(client), session(client)
    identifier, job = upload(client, owner)
    assert job["status"] == "complete"
    assert job["threshold"] == 0.43 and job["rows_processed"] == 2
    path = PREFIX + f"/analyses/{identifier}"
    rows = client.get(path + "/records", headers=owner).json()
    assert rows["records"][0]["rf"]["predicted_label"] == 1
    assert (
        rows["records"][0]["transaction_id"]
        == hashlib.sha256(CSV.encode()).hexdigest() + ":1"
    )
    detail = client.get(path + "/records/1", headers=owner).json()
    assert detail["derived"]["hour_of_day"] == 0
    assert list(detail["model_inputs"]) == FEATURE_COLUMNS
    direct = predict_records(
        bundle, pd.read_csv(io.StringIO(CSV))[SOURCE_COLUMNS], ["a", "b"]
    )
    assert detail["rf"]["score"] == direct.rf_risk_score.iloc[0]
    assert job["evaluation"] == evaluate_pair(
        [1, 0], [0.43, 0.1], [0.43, 0.7], 0.43, bundle.config.evaluation
    )
    for suffix in (
        "",
        "/records",
        "/records/1",
        "/exports/download",
        "/records/1/waterfall/rf",
    ):
        assert client.get(path + suffix, headers=other).status_code == 404
    assert (
        client.post(path + "/records/1/explanation", headers=other).status_code == 404
    )
    assert client.delete(path, headers=other).status_code == 404
    filtered = client.get(
        path + "/records?model=rf_smote&outcome=fp", headers=owner
    ).json()
    assert filtered["total"] == 1 and filtered["records"][0]["row_number"] == 2
    assert client.get(path, headers=owner).json()["evaluation"] == job["evaluation"]
    assert client.post(path + "/exports", headers=owner).status_code == 202
    exported = wait_for(
        lambda: client.get(path, headers=owner).json(),
        lambda j: j["export"]["status"] == "complete",
    )
    response = client.get(path + "/exports/download", headers=owner)
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        raw_name = next(n for n in archive.namelist() if n.startswith("Raw-Data_"))
        paper_name = raw_name.replace("Raw-Data_", "Experiment-Paper_").replace(
            ".csv", ".pdf"
        )
        assert set(archive.namelist()) == {raw_name, paper_name}
        assert archive.read(paper_name).startswith(b"%PDF-")
        metadata = exported["export"]["metadata"]
        assert metadata["scope"] == "uploaded_dataset"
        assert not metadata["held_out_membership_verified"]
        assert metadata["shap_coverage"]["computed_records"] == 0
        records = list(csv.DictReader(io.StringIO(archive.read(raw_name).decode())))
        assert len(records) == 2
        assert list(records[0]) == [
            "transaction_id",
            "upload_row_number",
            *HEADER.strip().split(","),
            "rf_score",
            "rf_predicted_label",
            "rf_smote_score",
            "rf_smote_predicted_label",
            "explanation_status",
            "rf_narrative",
            "rf_smote_narrative",
        ]
        source = list(csv.DictReader(io.StringIO(CSV)))
        for i, record in enumerate(records):
            assert record["upload_row_number"] == str(i + 1)
            assert {c: record[c] for c in source[i]} == source[i]
            assert float(record["rf_score"]) == direct.rf_risk_score.iloc[i]
            assert float(record["rf_smote_score"]) == direct.rf_smote_risk_score.iloc[i]
    folder = client.app.state.research.jobs[identifier]["folder"]
    assert client.delete(path, headers=owner).status_code == 204
    assert not folder.exists()


def test_paper_export_failure_can_retry(client, monkeypatch):
    from integritree.services.experiment_paper import PaperExportError

    owner = session(client)
    identifier, _ = upload(client, owner)
    path = PREFIX + f"/analyses/{identifier}"

    def failed(*args):
        raise PaperExportError("PDF converter is missing. Configure it and retry.")

    monkeypatch.setattr("integritree.services.experiment_paper.convert_pdf", failed)
    assert client.post(path + "/exports", headers=owner).status_code == 202
    job = wait_for(
        lambda: client.get(path, headers=owner).json(),
        lambda job: job["export"]["status"] == "failed",
    )
    assert "PDF converter is missing" in job["export"]["error"]
    folder = client.app.state.research.jobs[identifier]["folder"]
    service = client.app.state.research
    assert list(service.root.glob("session_*")) == [service.folder]
    assert not (folder / "results.zip").exists()
    assert not (folder / "results.partial.zip").exists()
    assert client.get(path + "/exports/download", headers=owner).status_code == 409
    monkeypatch.setattr(
        "integritree.services.experiment_paper.convert_pdf",
        lambda *args: b"%PDF-1.4\nretry\n%%EOF",
    )
    client.post(path + "/exports", headers=owner)
    wait_for(
        lambda: client.get(path, headers=owner).json(),
        lambda job: job["export"]["status"] == "complete",
    )
    assert client.get(path + "/exports/download", headers=owner).status_code == 200
    assert list(service.root.glob("session_*")) == [service.folder]


def test_repeated_exports_use_current_date_not_analysis_date(client, monkeypatch):
    owner = session(client)
    identifier, original = upload(client, owner)
    path = PREFIX + f"/analyses/{identifier}"
    current = datetime(2026, 10, 4, 15, 59, 59, tzinfo=timezone.utc)
    calls = []

    class ExportClock(datetime):
        @classmethod
        def now(cls, tz=None):
            calls.append(current)
            return current.astimezone(tz)

    monkeypatch.setattr("integritree.services.research.datetime", ExportClock)
    from integritree.services import research

    original_generate = research.generate_paper

    def crossing_midnight(*args):
        nonlocal current
        current = datetime.fromisoformat("2026-10-04T16:00:00+00:00")
        return original_generate(*args)

    monkeypatch.setattr(research, "generate_paper", crossing_midnight)
    for instant, date in (
        ("2026-10-04T15:59:59+00:00", "2026-10-04"),
        ("2026-10-04T16:00:00+00:00", "2026-10-05"),
    ):
        current = datetime.fromisoformat(instant)
        assert client.post(path + "/exports", headers=owner).status_code == 202
        job = wait_for(
            lambda: client.get(path, headers=owner).json(),
            lambda job: job["export"]["status"] == "complete",
        )
        response = client.get(path + "/exports/download", headers=owner)
        assert (
            'filename="integritree_results.zip"'
            in response.headers["content-disposition"]
        )
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            name = f"Experiment-Paper_{date}.pdf"
            assert sorted(archive.namelist()) == sorted([name, f"Raw-Data_{date}.csv"])
            assert archive.read(name).startswith(b"%PDF-")
        assert job["evaluation"] == original["evaluation"]
        assert job["completed_at"] == original["completed_at"]
    assert (
        len(calls) == 2
    )  # One date snapshot per export, including repeated downloads.


@pytest.mark.parametrize(
    "content,message,issue",
    [
        (HEADER + "1,TRANSFER,1,C_A,C_B,\n", "binary label", "isFraud"),
        (HEADER + "1,TRANSFER,1,C_A,C_B,2\n", "binary label", "isFraud"),
        (HEADER + "0,TRANSFER,-1,C_A,C_B,0\n", "predictor", "step"),
        (HEADER + "1,BAD,1,BAD,C_B,0\n", "predictor", "type"),
        (HEADER + "1,TRANSFER,1,C_A,0\n", "Data row 1", None),
        (HEADER + '1,TRANSFER,1,"C_A,C_B,0\n', "unexpected end", None),
        ("step,step,isFraud\n1,1,0\n", "Duplicate", None),
        ("step,isFraud\n1,0\n", "Missing columns", None),
        (HEADER, "at least one", None),
    ],
)
def test_invalid_upload_rejected_and_next_job_recovers(client, content, message, issue):
    owner = session(client)
    identifier, job = upload(client, owner, content)
    assert job["status"] == "failed" and message in job["error"]
    if issue:
        assert job["issues"][issue]["rows"] == [1]
    assert (
        client.get(
            PREFIX + f"/analyses/{identifier}/records", headers=owner
        ).status_code
        == 409
    )
    assert not (
        client.app.state.research.jobs[identifier]["folder"] / "records.sqlite"
    ).exists()
    assert upload(client, owner)[1]["status"] == "complete"


def test_pagination_search_missing_values(client):
    owner = session(client)
    identifier, job = upload(client, owner, HEADER + "1,,,C_A,C_B,0\n" * 23)
    assert job["status"] == "complete"
    path = PREFIX + f"/analyses/{identifier}/records"
    page = client.get(path + "?page=3", headers=owner).json()
    assert page["total"] == 23 and len(page["records"]) == 3
    search = client.get(path + "?search=:23", headers=owner).json()
    assert search["total"] == 1 and search["records"][0]["row_number"] == 23
    search = client.get(
        path + "?search=23&search_field=row_number", headers=owner
    ).json()
    assert search["total"] == 1 and search["records"][0]["row_number"] == 23
    fingerprint = page["records"][0]["transaction_id"].split(":")[0]
    assert (
        client.get(path, params={"search": fingerprint}, headers=owner).json()["total"]
        == 23
    )
    assert (
        client.get(
            path,
            params={"search": fingerprint, "search_field": "row_number"},
            headers=owner,
        ).json()["total"]
        == 0
    )
    assert (
        client.get(path + "?search=2&search_field=row_number", headers=owner).json()[
            "total"
        ]
        == 6
    )
    assert client.get(path + "?search_field=unknown", headers=owner).status_code == 422
    assert client.get(path + "?outcome=tp", headers=owner).status_code == 400


def test_limits_sessions_queue_refresh(client, monkeypatch):
    import integritree.api.routes.research as routes

    owner = session(client)
    assert (
        client.post(
            PREFIX + "/analyses?filename=x.csv",
            content=CSV,
            headers={"Content-Type": "text/csv"},
        ).status_code
        == 410
    )
    monkeypatch.setattr(routes, "MAX_BYTES", 20)
    for content in (CSV, iter([CSV.encode()])):
        assert (
            client.post(
                PREFIX + "/analyses?filename=x.csv",
                content=content,
                headers=owner | {"Content-Type": "text/csv"},
            ).status_code
            == 413
        )
    service = client.app.state.research
    first = service.reserve(owner["X-Research-Session"], "a.csv")
    second = service.reserve(owner["X-Research-Session"], "b.csv")
    with pytest.raises(ResearchError, match="queue is full"):
        service.reserve(owner["X-Research-Session"], "c.csv")
    service.abort_upload(first)
    service.abort_upload(second)
    monkeypatch.setattr(routes, "MAX_BYTES", 500 * 1024 * 1024)
    identifier, _ = upload(client, owner)
    assert (
        client.get(PREFIX + f"/analyses/{identifier}", headers=owner).json()["status"]
        == "complete"
    )


def test_restart_cleanup_boundaries(settings, experiment, bundle, tmp_path):
    service = ResearchService(settings, experiment, bundle)
    token = service.session()
    protected = tmp_path / "protected"
    protected.mkdir()
    with pytest.raises(ValueError, match="outside"):
        service._remove(protected)
    with pytest.raises(Exception, match="Another worker"):
        ResearchService(settings, experiment, bundle)
    folder = service.folder
    service.close()
    assert not folder.exists() and protected.exists()
    stale = service.root / "session_stale"
    stale.mkdir()
    (stale / "partial.csv").write_text("partial")
    replacement = ResearchService(settings, experiment, bundle)
    try:
        assert not stale.exists()
        with pytest.raises(ResearchError, match="Session expired"):
            replacement.check_session(token)
    finally:
        replacement.close()


@pytest.mark.parametrize(
    "kind", ["mixed", "no_positive", "all_positive", "ties", "inverse"]
)
def test_disk_evaluation_matches_existing_evaluator(experiment, kind):
    rng = np.random.default_rng(22)
    y = rng.integers(0, 2, 103)
    a, b = rng.random(103), rng.random(103)
    if kind == "no_positive":
        y[:] = 0
        a[:] = 0.1
        b[:] = 0.1
    if kind == "all_positive":
        y[:] = 1
    if kind == "ties":
        a = np.round(a, 1)
        b[:] = 0.43
    if kind == "inverse":
        a = 1 - y
        b = y
    db = sqlite3.connect(":memory:")
    db.execute(
        "CREATE TABLE records(label INTEGER,rf_score REAL,rf_pred INTEGER,rf_smote_score REAL,rf_smote_pred INTEGER)"
    )
    db.executemany(
        "INSERT INTO records VALUES(?,?,?,?,?)",
        [
            (int(v), float(r), int(r >= 0.43), float(s), int(s >= 0.43))
            for v, r, s in zip(y, a, b)
        ],
    )
    result = evaluate_database(db, 0.43, experiment.evaluation)
    direct = evaluate_pair(y, a, b, 0.43, experiment.evaluation)
    db.close()
    for model in ("rf", "rf_smote"):
        for name in result["models"][model]["metrics"]:
            assert result["models"][model]["metrics"][name]["value"] == pytest.approx(
                direct["models"][model]["metrics"][name]["value"]
            )
        assert (
            result["models"][model]["confusion_matrix"]
            == direct["models"][model]["confusion_matrix"]
        )
    assert result["statistical_test"] == direct["statistical_test"]


def test_csv_formula_safety():
    assert safe_csv(" =HYPERLINK('bad')").startswith("'")
    assert safe_csv("\tbad").startswith("'")
    assert safe_csv(-0.7) == -0.7
    assert safe_csv("C_NORMAL") == "C_NORMAL"


def test_late_batch_failure_removes_partial_records(client, monkeypatch):
    import integritree.services.research as research

    monkeypatch.setattr(research, "BATCH_SIZE", 1)
    owner = session(client)
    identifier, job = upload(client, owner, CSV + "3,TRANSFER,10,C_A,C_B,\n")
    assert job["status"] == "failed" and job["rows_processed"] == 2
    assert job["issues"]["isFraud"]["rows"] == [3]
    assert "evaluation" not in job
    assert not (
        client.app.state.research.jobs[identifier]["folder"] / "records.sqlite"
    ).exists()


def test_explanation_failure_retry_cache_and_export_coverage(client, tmp_path):
    service = client.app.state.research

    class Engine:
        calls = 0

        def explain(self, features, identities, analysis_id):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("Transient synthetic failure")
            outputs = []
            for model in ("rf", "rf_smote"):
                chart = tmp_path / f"{model}.svg"
                chart.write_text('<svg xmlns="http://www.w3.org/2000/svg"></svg>')
                outputs.append(
                    {
                        "model": model,
                        "waterfall_path": str(chart),
                        "narrative": "Synthetic explanation",
                        "transaction_id": identities[0],
                        "features": [
                            {
                                "feature": "hour_of_day",
                                "readable_value": "Simulation hour = 0",
                                "contribution": 0,
                            }
                        ],
                        "base_value": 0.43,
                        "output_value": 0.43,
                        "additivity_error": 0,
                        "top_positive_contributor": {
                            "status": "no_positive_contributor",
                            "feature": None,
                        },
                    }
                )
            return outputs

    engine = Engine()
    service.explanation_factory = lambda: engine
    owner = session(client)
    identifier, _ = upload(client, owner)
    path = PREFIX + f"/analyses/{identifier}"
    for expected in ("failed", "computed"):
        assert (
            client.post(path + "/records/1/explanation", headers=owner).status_code
            == 202
        )
        record = wait_for(
            lambda: client.get(path + "/records/1", headers=owner).json(),
            lambda r: r["explanation"]["status"] == expected,
        )
    assert (
        record["explanation"]["models"]["rf"]["top_positive_contributor"]["status"]
        == "no_positive_contributor"
    )
    assert "waterfall_path" not in record["explanation"]["models"]["rf"]
    assert (
        client.get(path + "/records/1/waterfall/rf", headers=owner).status_code == 200
    )
    original = client.get(path + "/records/1/waterfall/rf", headers=owner).content
    display = client.get(
        path + "/records/1/waterfall/rf?presentation=row_number", headers=owner
    )
    assert display.status_code == 200
    modal = client.get(
        path + "/records/1/waterfall/rf?presentation=row_number&layout=modal",
        headers=owner,
    )
    assert modal.status_code == 200
    assert "Reference" not in modal.text and "Benchmark RF | 1" not in modal.text
    assert "Hour of the Day" not in modal.text
    assert (
        client.get(
            path + "/records/1/waterfall/rf?layout=invalid", headers=owner
        ).status_code
        == 422
    )
    assert "Benchmark RF | 1" in display.text and "43.00%" in display.text
    fingerprint = hashlib.sha256(CSV.encode()).hexdigest()
    assert fingerprint not in display.text and fingerprint[:12] not in display.text
    assert (
        client.get(path + "/records/1/waterfall/rf", headers=owner).content == original
    )
    from integritree.ml.explainability import cached_contribution_chart

    assert "Contribution to Fraud Risk Score (Percentage Points)" in original.decode()
    assert (
        "SHAP contribution bar chart"
        in record["explanation"]["models"]["rf"]["chart_description"]
    )
    chart_path = cached_contribution_chart(
        record["explanation"]["models"]["rf"],
        service.jobs[identifier]["folder"],
        display_label="1",
    )
    rendered_at = chart_path.stat().st_mtime_ns
    assert (
        client.get(
            path + "/records/1/waterfall/rf?presentation=row_number", headers=owner
        ).content
        == display.content
    )
    assert chart_path.stat().st_mtime_ns == rendered_at
    assert (
        client.get(
            path + "/records/1/waterfall/rf?presentation=row_number",
            headers=session(client),
        ).status_code
        == 404
    )
    assert (
        client.post(path + "/records/1/explanation", headers=owner).json()["status"]
        == "computed"
    )
    assert engine.calls == 2
    client.post(path + "/exports", headers=owner)
    exported = wait_for(
        lambda: client.get(path, headers=owner).json(),
        lambda j: j["export"]["status"] == "complete",
    )
    assert exported["export"]["metadata"]["shap_coverage"]["computed_records"] == 1
    exported_zip = client.get(path + "/exports/download", headers=owner)
    with zipfile.ZipFile(io.BytesIO(exported_zip.content)) as archive:
        rows = list(
            csv.DictReader(
                io.StringIO(
                    archive.read(
                        next(n for n in archive.namelist() if n.startswith("Raw-Data_"))
                    ).decode()
                )
            )
        )
        assert rows[0]["transaction_id"] == fingerprint + ":1"


def test_raw_data_preserves_csv_formula_protection_and_precision(
    client, bundle, monkeypatch
):
    scores = np.array([0.12345678901234568, 0.9876543210987654])
    for model in bundle.models.values():
        monkeypatch.setattr(
            model, "predict_proba", lambda matrix: np.column_stack([1 - scores, scores])
        )
    owner = session(client)
    source = (
        HEADER.rstrip()
        + ",oldbalanceOrg\n1,TRANSFER,1.123456789012345,C_A,C_B,0,=1+1\n"
        + "2,PAYMENT,2.987654321098765,C_C,M_D,1, +SUM(1)\n"
    )
    identifier, _ = upload(client, owner, source, "<script>alert(1)</script>.csv")
    path = PREFIX + f"/analyses/{identifier}"
    client.post(path + "/exports", headers=owner)
    wait_for(
        lambda: client.get(path, headers=owner).json(),
        lambda j: j["export"]["status"] == "complete",
    )
    with zipfile.ZipFile(
        io.BytesIO(client.get(path + "/exports/download", headers=owner).content)
    ) as archive:
        assert len(archive.namelist()) == 2
        assert "report.html" not in archive.namelist()
        records = list(
            csv.DictReader(
                io.StringIO(
                    archive.read(
                        next(n for n in archive.namelist() if n.startswith("Raw-Data_"))
                    ).decode()
                )
            )
        )
        assert records[0]["oldbalanceOrg"] == "'=1+1"
        assert records[1]["oldbalanceOrg"] == "' +SUM(1)"
        assert [record["upload_row_number"] for record in records] == ["1", "2"]
        assert [record["amount"] for record in records] == [
            "1.123456789012345",
            "2.987654321098765",
        ]
        for i, record in enumerate(records):
            assert float(record["rf_score"]) == scores[i]
            assert float(record["rf_smote_score"]) == scores[i]
