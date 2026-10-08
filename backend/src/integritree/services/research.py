"""Temporary researcher analyses; no inference of held-out membership."""

import csv
import io
import json
import logging
import secrets
import shutil
import sqlite3
import threading
import uuid
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from integritree.ml import features as feature_module
from integritree.ml import inference, preprocessing
from integritree.ml.artifacts import load_bundle
from integritree.ml.data import file_sha256
from integritree.ml.features import (
    FEATURE_COLUMNS,
    SCALED_COLUMNS,
    SOURCE_COLUMNS,
    DataValidationError,
    engineer_features,
    validate_predictors,
)
from integritree.ml.inference import predict_features
from integritree.ml.staged_selection import run_lock
from integritree.services.experiment_paper import (
    PHILIPPINE_TIME,
    PaperExportError,
    generate_paper,
    paper_filename,
)
from integritree.services.research_metrics import evaluate_database

REQUIRED = SOURCE_COLUMNS + ["isFraud"]
PREPARED_REQUIRED = FEATURE_COLUMNS + ["isFraud"]
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
    def __init__(self, message, status=400, issues=None, code=None, layout_issues=None):
        super().__init__(message)
        self.status, self.issues = status, issues
        self.code = code
        self.layout_issues = layout_issues


def detect_input_format(headers):
    """Select a declared CSV layout; names do not prove preprocessing history."""
    layouts = (
        ("raw", REQUIRED, OPTIONAL),
        ("prepared", PREPARED_REQUIRED, []),
    )
    guidance = (
        " Accepted layouts: "
        + "; ".join(f"{name}: {', '.join(required)}" for name, required, _ in layouts)
        + ". Raw uploads may also contain the documented optional PaySim columns."
    )
    layout_issues = {
        "duplicate_columns": sorted(
            name for name, count in Counter(headers).items() if count > 1
        ),
        "formats": {
            name: {
                "missing_columns": sorted(set(required) - set(headers)),
                "unsupported_columns": sorted(set(headers) - set(required + optional)),
            }
            for name, required, optional in layouts
        },
    }
    if layout_issues["duplicate_columns"]:
        raise ResearchError(
            "Duplicate CSV column headers are not allowed." + guidance,
            code="unsupported_csv_layout",
            layout_issues=layout_issues,
        )
    problems = []
    for name, diagnostics in layout_issues["formats"].items():
        missing = diagnostics["missing_columns"]
        extra = diagnostics["unsupported_columns"]
        if not missing and not extra:
            return name
        problems.append(
            f"{name}: Missing columns: {', '.join(missing) or 'none'}. "
            f"Unsupported columns: {', '.join(extra) or 'none'}."
        )
    raise ResearchError(
        " ".join(problems) + guidance,
        code="unsupported_csv_layout",
        layout_issues=layout_issues,
    )


def validate_prepared_features(frame):
    """Validate model-ready values without transforming, imputing, or clipping."""
    features = frame.loc[:, FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    issues = {}

    def report(name, invalid):
        positions = np.flatnonzero(invalid)
        if len(positions):
            issues[name] = {
                "count": len(positions),
                "row_positions": positions[:20].tolist(),
            }

    for name in FEATURE_COLUMNS:
        values = features[name].to_numpy(dtype="float64")
        invalid = ~np.isfinite(values)
        if name not in SCALED_COLUMNS:
            invalid |= ~np.isin(values, [0, 1])
        report(name, invalid)
    types = [name for name in FEATURE_COLUMNS if name.startswith("type_")]
    report("transaction_type_indicators", (features[types] == 1).sum(axis=1) > 1)
    if issues:
        raise DataValidationError(issues)
    return features.astype("float64")


def prepared_display_features(features, state):
    """Undo scaling for display only; prediction always uses the supplied values."""
    derived = dict(features)
    for name, scale, offset in zip(state.scaled_columns, state.scale, state.offset):
        value = (derived[name] - offset) / scale
        # CSV round trips can leave an integer time value a few ulps away.
        if (
            name in ("hour_of_day", "day_of_week")
            and np.isfinite(value)
            and abs(value - round(value)) < 1e-9
        ):
            value = round(value)
        derived[name] = value if np.isfinite(value) else None
    return derived


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
            if job["input_format"] == "prepared":
                features = validate_prepared_features(frame)
            else:
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
                "Invalid predictor values. Prepared features must be finite numbers, with binary indicators and at most one active transaction type. Row numbers count data records, excluding the header."
                if job["input_format"] == "prepared"
                else "Invalid predictor values. Row numbers count data records, excluding the header.",
                issues=issues,
            ) from exc
        bundle = self.get_bundle()
        if job["input_format"] == "raw":
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
                    job["input_format"] = detect_input_format(headers)
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
            if getattr(exc, "layout_issues", None) is not None:
                job["layout_issues"] = exc.layout_issues
            job.update(
                status="failed",
                error=message,
                issues=getattr(exc, "issues", None),
                error_code=getattr(exc, "code", None),
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
            from integritree.ml.explainability import chart_description

            for item in result["explanation"].get("models", {}).values():
                item["chart_description"] = chart_description(item)
            result["original"] = json.loads(original)
            result["input_format"] = job["input_format"]
            result["model_inputs"] = dict(zip(FEATURE_COLUMNS, json.loads(features)))
            if job["input_format"] == "prepared":
                result["derived"] = prepared_display_features(
                    result["model_inputs"], self.get_bundle().preprocessor.state
                )
            else:
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

    def display_waterfall(
        self,
        job,
        number,
        model,
        explanation,
        *,
        presentation="row_number",
        layout="standalone",
    ):
        """Legacy route uses versioned contribution bars from stored SHAP values."""
        from integritree.ml.explainability import cached_contribution_chart

        return cached_contribution_chart(
            explanation,
            job["folder"],
            display_label=str(number) if presentation == "row_number" else None,
            layout=layout,
        )

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
            exported_at = datetime.now(timezone.utc)
            paper = generate_paper(
                job["evaluation"], self.root, self.settings.research_pdf_converter
            )
            with connect(job["folder"] / "records.sqlite") as db:
                covered = db.execute(
                    "SELECT count(*) FROM records WHERE explanation IS NOT NULL"
                ).fetchone()[0]
                bundle = self.get_bundle()
                metadata = {
                    "analysis_id": job["id"],
                    "filename": job["filename"],
                    "input_format": job["input_format"],
                    "preprocessing_applied": job["input_format"] == "raw",
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
                paper_name = paper_filename(exported_at)
                csv_name = f"Raw-Results_{exported_at.astimezone(PHILIPPINE_TIME):%Y-%m-%d}.csv"
                with zipfile.ZipFile(
                    output, "w", compression=zipfile.ZIP_DEFLATED
                ) as archive:
                    with archive.open(csv_name, "w", force_zip64=True) as binary:
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
                                ]
                            )
                            for row in db.execute(
                                "SELECT row_number, transaction_id, original, "
                                "rf_score, rf_pred, rf_smote_score, rf_smote_pred "
                                "FROM records ORDER BY row_number"
                            ):
                                (
                                    number,
                                    identity,
                                    original,
                                    rs,
                                    rp,
                                    ss,
                                    sp,
                                ) = row
                                original = json.loads(original)
                                # Prepared source cells have already passed numeric
                                # validation. Preserve signed values such as -0.25
                                # instead of escaping them as potential formulas.
                                source_values = [
                                    original[c]
                                    if job["input_format"] == "prepared"
                                    else safe_csv(original[c])
                                    for c in job["source_columns"]
                                ]
                                writer.writerow(
                                    [
                                        safe_csv(identity),
                                        number,
                                        *source_values,
                                    ]
                                    + [
                                        safe_csv(v)
                                        for v in [
                                            rs,
                                            rp,
                                            ss,
                                            sp,
                                        ]
                                    ]
                                )
                    archive.writestr(paper_name, paper)
                output.replace(job["folder"] / "results.zip")
            with self.lock:
                job["export"] = {"status": "complete", "metadata": metadata}
        except Exception as exc:
            LOG.exception("Export failed")
            (job["folder"] / "results.partial.zip").unlink(missing_ok=True)
            with self.lock:
                job["export"] = {
                    "status": "failed",
                    "error": str(exc)
                    if isinstance(exc, PaperExportError)
                    else "Export failed. Retry the download.",
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
