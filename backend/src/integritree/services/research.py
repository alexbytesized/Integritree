"""Temporary researcher analyses; no inference of held-out membership."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
import csv
import html
import io
import json
import logging
from pathlib import Path
import secrets
import shutil
import sqlite3
import threading
import uuid
import zipfile
import pandas as pd
from integritree.ml.artifacts import load_bundle
from integritree.ml.data import file_sha256
from integritree.ml.features import (
    SOURCE_COLUMNS,
    FEATURE_COLUMNS,
    DataValidationError,
    validate_predictors,
    engineer_features,
)
from integritree.ml.inference import predict_features
from integritree.ml import inference, features as feature_module, preprocessing
from integritree.ml.staged_selection import run_lock
from integritree.services.research_metrics import evaluate_database

REQUIRED = SOURCE_COLUMNS + ["isFraud"]
OPTIONAL = [
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "isFlaggedFraud",
]
BATCH_SIZE = 25_000
MAX_BYTES = 500 * 1024 * 1024
LOG = logging.getLogger(__name__)


class ResearchError(Exception):
    def __init__(self, message, status=400, issues=None):
        super().__init__(message)
        self.status, self.issues = status, issues


@contextmanager
def connect(path):
    db = sqlite3.connect(path, timeout=30)
    db.execute("PRAGMA cache_size=-8192")
    db.execute("PRAGMA temp_store=FILE")
    try:
        with db:
            yield db
    finally:
        db.close()


def safe_csv(value):
    if isinstance(value, str) and (
        value.startswith(("\t", "\r", "\n"))
        or value.lstrip().startswith(("=", "+", "-", "@"))
    ):
        return "'" + value
    return value


class ResearchService:
    def __init__(self, settings, experiment, bundle=None, explanation_factory=None):
        self.settings, self.experiment = settings, experiment
        self.root = (settings.backend_root / "runtime/research_sessions").resolve()
        runtime = (settings.backend_root / "runtime").resolve()
        if not self.root.is_relative_to(runtime) or self.root == runtime:
            raise ValueError("Research runtime must remain inside backend/runtime")
        self.root.mkdir(parents=True, exist_ok=True)
        self.process_lock = run_lock(self.root)
        self.process_lock.__enter__()
        for path in self.root.glob("session_*"):
            self._remove(path)
        self.folder = self.root / ("session_" + uuid.uuid4().hex)
        self.folder.mkdir()
        self.sessions, self.jobs = set(), {}
        self.lock, self.model_lock = threading.RLock(), threading.Lock()
        self.slots, self.aux_slots = (
            threading.BoundedSemaphore(2),
            threading.BoundedSemaphore(24),
        )
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="research")
        self.aux = ThreadPoolExecutor(max_workers=1, thread_name_prefix="research-aux")
        self.bundle, self.engine = bundle, None
        self.explanation_factory, self.closed = explanation_factory, False

    def _remove(self, path):
        resolved = path.resolve()
        if (
            path.is_symlink()
            or not resolved.is_relative_to(self.root)
            or resolved == self.root
        ):
            raise ValueError("Refusing cleanup outside research session runtime")
        if path.exists():
            shutil.rmtree(path)

    def close(self):
        self.closed = True
        self.worker.shutdown(wait=True)
        self.aux.shutdown(wait=True)
        self._remove(self.folder)
        self.process_lock.__exit__(None, None, None)

    def session(self):
        token = secrets.token_urlsafe(32)
        with self.lock:
            self.sessions.add(token)
        return token

    def check_session(self, token):
        if not token or token not in self.sessions or self.closed:
            raise ResearchError(
                "Session expired. Upload the CSV again to start a new analysis.", 410
            )

    def get_bundle(self):
        with self.model_lock:
            if self.bundle is None:
                bundle = load_bundle(self.settings.research_model_dir)
                if bundle.metadata.get("stage") != "validation_selected":
                    raise ValueError(
                        "Research app requires a frozen selected model bundle"
                    )
                self.bundle = bundle
            return self.bundle

    def reserve(self, token, filename):
        self.check_session(token)
        if not filename.lower().endswith(".csv"):
            raise ResearchError("Upload a UTF-8 CSV file.")
        if not self.slots.acquire(blocking=False):
            raise ResearchError(
                "Analysis queue is full: one running and one queued upload are allowed. Retry shortly.",
                429,
            )
        with self.lock:
            identifier = uuid.uuid4().hex
            folder = self.folder / identifier
            folder.mkdir()
            job = {
                "id": identifier,
                "owner": token,
                "folder": folder,
                "filename": filename.replace("\\", "/").split("/")[-1][:255],
                "status": "uploading",
                "rows_processed": 0,
                "bytes_received": 0,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "explanations": {},
                "export": {"status": "not_requested"},
                "aux_pending": 0,
            }
            self.jobs[identifier] = job
            return job

    def abort_upload(self, job):
        with self.lock:
            self.jobs.pop(job["id"], None)
            self._remove(job["folder"])
        self.slots.release()

    def submit(self, job, fingerprint):
        job.update(status="queued", upload_sha256=fingerprint)
        self.worker.submit(self._run, job)

    def owned(self, token, identifier, ready=False):
        self.check_session(token)
        with self.lock:
            job = self.jobs.get(identifier)
            if not job or job["owner"] != token:
                raise ResearchError("Analysis not found in this session.", 404)
            if ready and job["status"] != "complete":
                raise ResearchError("Analysis is not complete.", 409)
            return job

    def status(self, token, identifier):
        job = self.owned(token, identifier)
        with self.lock:
            return {
                k: v
                for k, v in job.items()
                if k not in ("owner", "folder", "explanations", "aux_pending")
            }

    def _batch(self, db, job, headers, rows, start):
        frame = pd.DataFrame(rows, columns=headers)
        labels = frame.isFraud.str.strip()
        invalid = ~labels.isin(["0", "1"])
        if invalid.any():
            raise ResearchError(
                "isFraud must contain a complete binary label (0 or 1).",
                issues={
                    "isFraud": {
                        "count": int(invalid.sum()),
                        "rows": (invalid[invalid].index[:20] + start).tolist(),
                    }
                },
            )
        try:
            inputs = validate_predictors(frame[SOURCE_COLUMNS])
        except DataValidationError as exc:
            issues = {
                name: {
                    "count": info["count"],
                    "rows": [start + i for i in info["row_positions"]],
                }
                for name, info in exc.issues.items()
            }
            raise ResearchError(
                "Invalid predictor values. Row numbers count data records, excluding the header.",
                issues=issues,
            ) from exc
        bundle = self.get_bundle()
        features = bundle.preprocessor.transform(inputs)
        identities = [
            f"{job['upload_sha256']}:{i}" for i in range(start, start + len(frame))
        ]
        predictions = predict_features(
            bundle.models,
            features,
            identities,
            bundle.config.scoring.threshold,
            bundle.metadata["run_id"],
        )
        source, feature_rows = frame.to_dict("records"), features.to_numpy().tolist()
        db.executemany(
            "INSERT INTO records VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                (
                    start + i,
                    identities[i],
                    json.dumps(source[i], ensure_ascii=False),
                    json.dumps(feature_rows[i]),
                    int(labels.iloc[i]),
                    float(p.rf_risk_score),
                    int(p.rf_predicted_label),
                    float(p.rf_smote_risk_score),
                    int(p.rf_smote_predicted_label),
                    None,
                )
                for i, p in enumerate(predictions.itertuples(index=False))
            ),
        )
        db.commit()
        job["rows_processed"] += len(frame)

    def _run(self, job):
        try:
            job["status"] = "loading_models"
            bundle = self.get_bundle()
            job.update(
                status="processing",
                threshold=bundle.config.scoring.threshold,
                model_run_id=bundle.metadata["run_id"],
            )
            with connect(job["folder"] / "records.sqlite") as db:
                db.execute(
                    "CREATE TABLE records (row_number INTEGER PRIMARY KEY, transaction_id TEXT UNIQUE, original TEXT, features TEXT, label INTEGER, rf_score REAL, rf_pred INTEGER, rf_smote_score REAL, rf_smote_pred INTEGER, explanation TEXT)"
                )
                with (job["folder"] / "input.csv").open(
                    encoding="utf-8-sig", newline=""
                ) as stream:
                    reader = csv.reader(stream, strict=True)
                    headers = next(reader, [])
                    if len(headers) != len(set(headers)):
                        raise ResearchError(
                            "Duplicate CSV column headers are not allowed."
                        )
                    missing, extra = (
                        sorted(set(REQUIRED) - set(headers)),
                        sorted(set(headers) - set(REQUIRED + OPTIONAL)),
                    )
                    if missing or extra:
                        raise ResearchError(
                            f"Missing columns: {', '.join(missing) or 'none'}. Unsupported columns: {', '.join(extra) or 'none'}."
                        )
                    job["source_columns"] = headers
                    rows, start, batch_chars = [], 1, 0
                    for number, row in enumerate(reader, 1):
                        if self.closed:
                            raise ResearchError("Backend is shutting down.")
                        if len(row) != len(headers):
                            raise ResearchError(
                                f"Data row {number}: expected {len(headers)} columns, received {len(row)}."
                            )
                        rows.append(row)
                        batch_chars += sum(map(len, row))
                        if len(rows) == BATCH_SIZE or batch_chars >= 8 * 1024 * 1024:
                            self._batch(db, job, headers, rows, start)
                            start += len(rows)
                            rows = []
                            batch_chars = 0
                    if rows:
                        self._batch(db, job, headers, rows, start)
                if not job["rows_processed"]:
                    raise ResearchError("CSV must contain at least one data record.")
                job["status"] = "evaluating"
                job["evaluation"] = evaluate_database(
                    db, job["threshold"], self.experiment.evaluation
                )
            job.update(
                status="complete", completed_at=datetime.now(timezone.utc).isoformat()
            )
        except Exception as exc:
            LOG.exception("Research analysis %s failed", job["id"])
            message = (
                str(exc)
                if isinstance(exc, (ResearchError, UnicodeError, csv.Error))
                else "Analysis failed. Check backend logs and retry."
            )
            if isinstance(exc, csv.Error):
                message = f"Malformed CSV near physical line {reader.line_num}: {exc}"
            job.update(
                status="failed", error=message, issues=getattr(exc, "issues", None)
            )
            for name in ("records.sqlite", "records.sqlite-journal"):
                (job["folder"] / name).unlink(missing_ok=True)
        finally:
            (job["folder"] / "input.csv").unlink(missing_ok=True)
            self.slots.release()

    def _record(self, job, row, detail=False):
        number, identity, original, features, label, rs, rp, ss, sp, explanation = row
        result = {
            "row_number": number,
            "transaction_id": identity,
            "actual_label": label,
            "rf": {"score": rs, "predicted_label": rp},
            "rf_smote": {"score": ss, "predicted_label": sp},
            "explanation": json.loads(explanation)
            if explanation
            else job["explanations"].get(number, {"status": "not_requested"}),
        }
        if detail:
            result["original"] = json.loads(original)
            result["model_inputs"] = dict(zip(FEATURE_COLUMNS, json.loads(features)))
            inputs = pd.DataFrame([result["original"]])[SOURCE_COLUMNS]
            result["derived"] = (
                engineer_features(
                    inputs, self.get_bundle().preprocessor.state.amount_median
                )
                .iloc[0]
                .to_dict()
            )
            result["threshold"] = job["threshold"]
        return result

    def records(
        self,
        token,
        identifier,
        page=1,
        search="",
        model="both",
        outcome="all",
        search_field="transaction_id",
    ):
        job = self.owned(token, identifier, True)
        if search_field not in ("transaction_id", "row_number"):
            raise ResearchError("Unknown transaction search field.")
        column = (
            "CAST(row_number AS TEXT)"
            if search_field == "row_number"
            else "transaction_id"
        )
        where, args = f"instr({column}, ?) > 0", [search.strip()]
        if outcome != "all":
            if model not in ("rf", "rf_smote"):
                raise ResearchError("Choose a model before an outcome filter.")
            label, pred = {"tp": (1, 1), "fp": (0, 1), "tn": (0, 0), "fn": (1, 0)}[
                outcome
            ]
            where += f" AND label=? AND {model}_pred=?"
            args += [label, pred]
        with connect(job["folder"] / "records.sqlite") as db:
            total = db.execute(
                "SELECT count(*) FROM records WHERE " + where, args
            ).fetchone()[0]
            page = min(page, max(1, (total + 9) // 10))
            rows = db.execute(
                "SELECT * FROM records WHERE "
                + where
                + " ORDER BY row_number LIMIT 10 OFFSET ?",
                args + [(page - 1) * 10],
            ).fetchall()
        return {
            "records": [self._record(job, r) for r in rows],
            "total": total,
            "page": page,
            "page_size": 10,
        }

    def display_waterfall(self, job, number, model, explanation):
        """Cache a presentation-only chart; keep canonical IDs and artifacts intact."""
        from integritree.ml.explainability import waterfall

        target = job["folder"] / f"{number}_{model}_display_v1.svg"
        if not target.exists():
            temporary = job["folder"] / f"{uuid.uuid4().hex}.svg"
            try:
                waterfall(explanation, temporary, display_label=str(number))
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
        return target

    def record(self, token, identifier, number):
        job = self.owned(token, identifier, True)
        with connect(job["folder"] / "records.sqlite") as db:
            row = db.execute(
                "SELECT * FROM records WHERE row_number=?", (number,)
            ).fetchone()
        if row is None:
            raise ResearchError("Record not found.", 404)
        return self._record(job, row, True)

    def _schedule(self, job, function, *args):
        if not self.aux_slots.acquire(blocking=False):
            raise ResearchError("Explanation/export queue is full. Retry shortly.", 429)
        job["aux_pending"] += 1

        def work():
            try:
                function(job, *args)
            finally:
                with self.lock:
                    job["aux_pending"] -= 1
                self.aux_slots.release()

        self.aux.submit(work)

    def explain(self, token, identifier, number):
        with self.lock:
            job = self.owned(token, identifier, True)
            current = self.record(token, identifier, number)["explanation"]
            if current["status"] in ("computed", "pending"):
                return current
            self._schedule(job, self._explain, number)
            job["explanations"][number] = {"status": "pending"}
            return {"status": "pending"}

    def _explain(self, job, number):
        try:
            if self.engine is None:
                if self.explanation_factory:
                    self.engine = self.explanation_factory()
                else:
                    from integritree.ml.explainability import ExplanationEngine

                    self.engine = ExplanationEngine(
                        self.settings.research_model_dir,
                        self.settings.research_prepared_dir,
                        self.experiment.shap,
                        self.folder / "shap",
                        bundle=self.get_bundle(),
                    )
            with connect(job["folder"] / "records.sqlite") as db:
                identity, features = db.execute(
                    "SELECT transaction_id, features FROM records WHERE row_number=?",
                    (number,),
                ).fetchone()
                self.engine.cache_root = job["folder"] / "shap"
                values = self.engine.explain(
                    pd.DataFrame([json.loads(features)], columns=FEATURE_COLUMNS),
                    [identity],
                    job["id"],
                )
                models = {}
                for value in values:
                    item = dict(value)
                    svg = Path(item.pop("waterfall_path"))
                    shutil.copyfile(
                        svg, job["folder"] / f"{number}_{item['model']}.svg"
                    )
                    item["waterfall_url"] = (
                        f"/api/v1/research/analyses/{job['id']}/records/{number}/waterfall/{item['model']}"
                    )
                    models[item["model"]] = item
                db.execute(
                    "UPDATE records SET explanation=? WHERE row_number=?",
                    (json.dumps({"status": "computed", "models": models}), number),
                )
            with self.lock:
                job["explanations"].pop(number, None)
        except Exception:
            LOG.exception("Explanation failed")
            with self.lock:
                job["explanations"][number] = {
                    "status": "failed",
                    "error": "Explanation failed; retry is available.",
                }

    def export(self, token, identifier):
        with self.lock:
            job = self.owned(token, identifier, True)
            if job["export"]["status"] == "pending":
                return job["export"]
            self._schedule(job, self._export)
            job["export"] = {"status": "pending"}
            return job["export"]

    def _export(self, job):
        try:
            with connect(job["folder"] / "records.sqlite") as db:
                covered = db.execute(
                    "SELECT count(*) FROM records WHERE explanation IS NOT NULL"
                ).fetchone()[0]
                bundle = self.get_bundle()
                metadata = {
                    "analysis_id": job["id"],
                    "filename": job["filename"],
                    "upload_sha256": job["upload_sha256"],
                    "scope": "uploaded_dataset",
                    "held_out_membership_verified": False,
                    "same_record_pairing_verified": True,
                    "rows": job["rows_processed"],
                    "actual_fraud": job["evaluation"]["models"]["rf"]["actual_fraud"],
                    "actual_legitimate": job["rows_processed"]
                    - job["evaluation"]["models"]["rf"]["actual_fraud"],
                    "model_run_id": job["model_run_id"],
                    "threshold": job["threshold"],
                    "evaluation_timestamp": job["completed_at"],
                    "evaluation_policy": self.experiment.evaluation.model_dump(),
                    "feature_order": FEATURE_COLUMNS,
                    "preprocessing": bundle.preprocessor.state.model_dump(),
                    "random_forest": bundle.config.random_forest.model_dump(),
                    "smote": bundle.config.smote.model_dump(),
                    "shap_policy": self.experiment.shap.model_dump(),
                    "shap_coverage": {
                        "computed_records": covered,
                        "total_records": job["rows_processed"],
                        "models_per_record": 2,
                    },
                    "retention": "temporary; downloaded files are the retained copies",
                }
                metadata["model_package_versions"] = bundle.metadata.get(
                    "package_versions"
                )
                metadata["shap_explainer_identity"] = (
                    getattr(self.engine, "identity", None) if covered else None
                )
                metadata["implementation_sha256"] = {
                    p.name: file_sha256(p)
                    for p in (
                        Path(__file__),
                        Path(__file__).with_name("research_metrics.py"),
                        Path(inference.__file__),
                        Path(feature_module.__file__),
                        Path(preprocessing.__file__),
                    )
                }
                model_meta = self.settings.research_model_dir / "metadata.json"
                if model_meta.exists():
                    metadata["model_metadata_sha256"] = file_sha256(model_meta)
                output = job["folder"] / "results.partial.zip"
                with zipfile.ZipFile(
                    output, "w", compression=zipfile.ZIP_DEFLATED
                ) as archive:
                    archive.writestr(
                        "metadata.json", json.dumps(metadata, indent=2, allow_nan=False)
                    )
                    archive.writestr(
                        "evaluation.json",
                        json.dumps(job["evaluation"], indent=2, allow_nan=False),
                    )
                    with archive.open("results.csv", "w", force_zip64=True) as binary:
                        with io.TextIOWrapper(
                            binary, encoding="utf-8", newline=""
                        ) as text:
                            writer = csv.writer(text)
                            writer.writerow(
                                [
                                    "transaction_id",
                                    "upload_row_number",
                                    *job["source_columns"],
                                    "rf_score",
                                    "rf_predicted_label",
                                    "rf_smote_score",
                                    "rf_smote_predicted_label",
                                    "explanation_status",
                                    "rf_narrative",
                                    "rf_smote_narrative",
                                ]
                            )
                            for row in db.execute(
                                "SELECT * FROM records ORDER BY row_number"
                            ):
                                (
                                    number,
                                    identity,
                                    original,
                                    _,
                                    _,
                                    rs,
                                    rp,
                                    ss,
                                    sp,
                                    explanation,
                                ) = row
                                original = json.loads(original)
                                explanation = (
                                    json.loads(explanation) if explanation else {}
                                )
                                models = explanation.get("models", {})
                                writer.writerow(
                                    [
                                        safe_csv(v)
                                        for v in [
                                            identity,
                                            number,
                                            *[
                                                original[c]
                                                for c in job["source_columns"]
                                            ],
                                            rs,
                                            rp,
                                            ss,
                                            sp,
                                            explanation.get(
                                                "status",
                                                job["explanations"]
                                                .get(number, {})
                                                .get("status", "not_requested"),
                                            ),
                                            models.get("rf", {}).get("narrative", ""),
                                            models.get("rf_smote", {}).get(
                                                "narrative", ""
                                            ),
                                        ]
                                    ]
                                )
                    report = "<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Integritree analysis</title><style>body{font:16px system-ui;margin:2rem auto;padding:1rem;max-width:70rem;color:#123}table{border-collapse:collapse;width:100%}td,th{padding:.6rem;border:1px solid #ccd;text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f7fa;padding:1rem}h1,h2{color:#086dc1}</style><body><h1>Uploaded dataset evaluation</h1>"
                    report += "<p>Held-out membership was not verified. Metrics cover the complete upload, irrespective of table filters.</p>"
                    report += f"<p>File: {html.escape(job['filename'])}. Records: {job['rows_processed']:,}. Shared threshold: {job['threshold']:.0%}.</p>"
                    report += "<h2>Model metrics (coefficients on their original scale)</h2><table><tr><th>Metric</th><th>RF</th><th>RF-SMOTE</th></tr>"
                    for metric_name in (
                        "precision",
                        "recall",
                        "f1",
                        "mcc",
                        "pr_auc",
                        "accuracy",
                    ):
                        report += (
                            "<tr><th>"
                            + (
                                "PR-AUC (Average Precision)"
                                if metric_name == "pr_auc"
                                else metric_name.upper()
                            )
                            + "</th>"
                        )
                        for model in ("rf", "rf_smote"):
                            measured = job["evaluation"]["models"][model]["metrics"][
                                metric_name
                            ]
                            report += (
                                "<td>"
                                + html.escape(
                                    str(measured["value"])
                                    if measured["value"] is not None
                                    else "Unavailable: " + measured["reason"]
                                )
                                + "</td>"
                            )
                        report += "</tr>"
                    report += (
                        "</table><h2>Confusion matrices, comparisons and McNemar test</h2><pre>"
                        + html.escape(json.dumps(job["evaluation"], indent=2))
                        + "</pre>"
                    )
                    report += (
                        "<h2>Provenance and explanation coverage</h2><pre>"
                        + html.escape(json.dumps(metadata, indent=2))
                        + "</pre></body></html>"
                    )
                    archive.writestr("report.html", report)
                output.replace(job["folder"] / "results.zip")
            with self.lock:
                job["export"] = {"status": "complete", "metadata": metadata}
        except Exception:
            LOG.exception("Export failed")
            with self.lock:
                job["export"] = {
                    "status": "failed",
                    "error": "Export failed. Retry the download.",
                }

    def delete(self, token, identifier):
        with self.lock:
            job = self.owned(token, identifier)
            if job["status"] not in ("complete", "failed") or job["aux_pending"]:
                raise ResearchError(
                    "Wait for this analysis and its explanation/export jobs to finish before clearing it.",
                    409,
                )
            self._remove(job["folder"])
            self.jobs.pop(identifier)
