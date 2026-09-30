"""Session-owned local image jobs, confirmations, paired results and temporary exports."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import secrets
import shutil
import subprocess
import threading
import uuid
import zipfile

from PIL import Image
from pydantic import ValidationError

from integritree.ml.data import file_sha256
from integritree.ml.staged_selection import run_lock
from integritree.receipts.audit import load_image, MAX_BYTES
from integritree.receipts.contracts import (
    ReceiptSource, ExtractedReceiptFields, ConfirmedReceiptFields, confirm_receipt, LAYOUT_WORKFLOWS,
)
from integritree.receipts.layouts import identify_layout
from integritree.receipts.mapping import predict_receipt


class ReceiptError(Exception):
    def __init__(self, message, status=400, issues=None):
        super().__init__(message)
        self.status, self.issues = status, issues


class ReceiptService:
    def __init__(self, settings, experiment, bundle_provider, extractor=None, explanation_factory=None):
        self.settings, self.experiment = settings, experiment
        self.bundle_provider, self.extractor = bundle_provider, extractor or self._extract
        self.explanation_factory = explanation_factory
        self.root = (settings.backend_root / "runtime/receipt_sessions").resolve()
        runtime = (settings.backend_root / "runtime").resolve()
        if not self.root.is_relative_to(runtime) or self.root == runtime:
            raise ValueError("Receipt storage must stay inside backend/runtime")
        self.root.mkdir(parents=True, exist_ok=True)
        self.process_lock = run_lock(self.root)
        self.process_lock.__enter__()
        for path in self.root.glob("session_*"):
            self._remove(path)
        self.folder = self.root / ("session_" + uuid.uuid4().hex)
        self.folder.mkdir()
        self.lock = threading.RLock()
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="receipts")
        self.slots = threading.BoundedSemaphore(2)
        self.sessions, self.jobs = set(), {}
        self.closed, self.engine = False, None

    def _remove(self, path):
        resolved = path.resolve()
        if path.is_symlink() or not resolved.is_relative_to(self.root) or resolved == self.root:
            raise ValueError("Refusing cleanup outside receipt sessions")
        if path.exists():
            shutil.rmtree(path)

    def close(self):
        self.closed = True
        self.worker.shutdown(wait=True)
        self._remove(self.folder)
        self.process_lock.__exit__(None, None, None)

    def session(self):
        with self.lock:
            if self.closed:
                raise ReceiptError("Backend is shutting down.", 503)
            token = secrets.token_urlsafe(32)
            self.sessions.add(token)
            return token

    def check_session(self, token):
        if self.closed or not token or token not in self.sessions:
            raise ReceiptError("Session expired. Upload the receipt again.", 410)

    def owned(self, token, identifier):
        self.check_session(token)
        job = self.jobs.get(identifier)
        if not job or job["owner"] != token:
            raise ReceiptError("Receipt not found in this session.", 404)
        return job

    def reserve(self, token, filename):
        with self.lock:
            self.check_session(token)
            # One retained receipt per browser session; Clear before replacing it.
            if any(j["owner"] == token for j in self.jobs.values()):
                raise ReceiptError("Clear the existing receipt before uploading another.", 409)
            if not self.slots.acquire(blocking=False):
                raise ReceiptError("Receipt queue is full. Retry shortly.", 429)
            identifier = uuid.uuid4().hex
            folder = self.folder / identifier
            folder.mkdir()
            job = dict(id=identifier, owner=token, folder=folder, filename=filename.replace("\\", "/").split("/")[-1],
                       status="uploading", busy=True, bytes_received=0, revision=0, result=None, confirmation=None,
                       source=None, error=None, created_at=datetime.now(timezone.utc).isoformat())
            self.jobs[identifier] = job
            return job

    def abort(self, job):
        with self.lock:
            self.jobs.pop(job["id"], None)
            self._remove(job["folder"])
            self.slots.release()

    def submit(self, job, digest):
        with self.lock:
            job.update(status="queued", image_sha256=digest)
            self.worker.submit(self._task, job, self._ocr)

    def _task(self, job, function):
        try:
            function(job)
        except Exception:
            # Do not log receipt contents, model input values or raw exception messages.
            with self.lock:
                job.update(status="failed", error="Receipt processing failed. Retry or clear this receipt.")
        finally:
            with self.lock:
                job["busy"] = False
            self.slots.release()

    def _schedule(self, job, function):
        if job["busy"]:
            raise ReceiptError("Wait for the current receipt job to finish.", 409)
        if not self.slots.acquire(blocking=False):
            raise ReceiptError("Receipt queue is full. Retry shortly.", 429)
        job["busy"] = True
        self.worker.submit(self._task, job, function)

    def _extract(self, path):
        python = self.settings.receipt_ocr_python
        if not python.is_file():
            raise ReceiptError("Local OCR is not installed. Follow the receipt setup guide, then retry.", 503)
        output = path.parent / "ocr.json"
        output.unlink(missing_ok=True)
        script = Path(__file__).resolve().parents[3] / "scripts/extract_receipt.py"
        subprocess.run([str(python), str(script), str(path), str(output)], check=True, timeout=120,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return json.loads(output.read_text(encoding="utf-8"))

    def _ocr(self, job):
        with self.lock:
            job.update(status="extracting", error=None)
        path = job["folder"] / "image"
        try:
            with load_image(path):
                pass
            with Image.open(path) as image:
                media_type = "image/png" if image.format == "PNG" else "image/jpeg"
        except (ValueError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning):
            with self.lock:
                job.update(status="unsupported", error="Use a valid single PNG/JPEG, at most 10 MiB and 20 million pixels.")
            return
        try:
            extraction = self.extractor(path)
        except ReceiptError as exc:
            with self.lock:
                job.update(status="failed", error=str(exc))
            return
        layout, observed, error = identify_layout(extraction["lines"])
        with self.lock:
            job["media_type"] = media_type
            if not layout:
                job.update(status="unsupported", error=error)
                return
            candidates = extraction["parsed"]["fields"]
            job["source"] = ReceiptSource(
                analysis_id=uuid.UUID(hex=job["id"]), image_id=uuid.uuid4(), image_sha256=job["image_sha256"],
                media_type=media_type, layout_id=layout, extractor_version=extraction["extractor_version"],
                parser_version=extraction["parsed"]["parser_version"],
                extracted=ExtractedReceiptFields(**candidates), observed_destination_role=observed)
            job.update(status="awaiting_confirmation", error=None)

    def status(self, token, identifier):
        with self.lock:
            job = self.owned(token, identifier)
            result = {k: job[k] for k in ("id", "filename", "status", "busy", "revision", "error", "created_at")}
            if job["source"]:
                result["workflow"] = LAYOUT_WORKFLOWS[job["source"].layout_id]
                result["candidates"] = job["source"].extracted.model_dump()
            if job["confirmation"]:
                result["confirmed_fields"] = job["confirmation"].fields.model_dump(mode="json")
            if job["result"]:
                result["result"] = job["result"]
            return result

    def confirm(self, token, identifier, fields, confirmed, expected_revision):
        with self.lock:
            job = self.owned(token, identifier)
            if job["busy"] or job["source"] is None:
                raise ReceiptError("This receipt is not ready for confirmation.", 409)
            if expected_revision != job["revision"]:
                raise ReceiptError("Receipt changed. Reload before confirming.", 409)
            if file_sha256(job["folder"] / "image") != job["source"].image_sha256:
                raise ReceiptError("Stored image changed. Clear and upload it again.", 409)
            try:
                confirmation = confirm_receipt(job["source"], ConfirmedReceiptFields.model_validate(fields),
                                               confirmed=confirmed, previous=job["confirmation"])
            except ValidationError as exc:
                raise ReceiptError("Check the receipt fields and confirm the supported account roles.", 422,
                                   exc.errors(include_input=False, include_context=False, include_url=False)) from None
            if not self.slots.acquire(blocking=False):
                raise ReceiptError("Receipt queue is full. Retry shortly.", 429)
            # Retire old derived artifacts before accepting the next revision.
            try:
                for path in job["folder"].glob("revision_*"):
                    self._remove(path)
            except Exception:
                self.slots.release()
                raise ReceiptError("Could not retire the previous result. Clear this receipt and try again.", 503) from None
            job.update(confirmation=confirmation, revision=confirmation.revision, result=None,
                       status="predicting", error=None, busy=True)
            self.worker.submit(self._task, job, self._predict)
            return {"id": identifier, "revision": job["revision"], "status": job["status"]}

    def _predict(self, job):
        bundle = self.bundle_provider()
        result = predict_receipt(bundle, job["confirmation"])
        revision_dir = job["folder"] / f"revision_{job['revision']}"
        revision_dir.mkdir(exist_ok=True)
        row = result.predictions.iloc[0]
        record = dict(analysis_id=job["id"], revision=job["revision"], input_sha256=result.input_sha256,
                      mapping_version=result.receipt.mapping_version, model_run_id=bundle.metadata["run_id"],
                      threshold=bundle.config.scoring.threshold, original=result.receipt.fields.model_dump(mode="json"),
                      derived=result.unscaled_features.iloc[0].to_dict(), model_inputs=result.scaled_features.iloc[0].to_dict(),
                      field_provenance=result.receipt.field_provenance, explanation={"status": "pending"})
        for name in ("rf", "rf_smote"):
            record[name] = {"score": float(row[f"{name}_risk_score"]), "predicted_label": int(row[f"{name}_predicted_label"])}
        with self.lock:
            job.update(result=record, status="explaining")
        self._explain(job, result.scaled_features)
        with self.lock:
            job["status"] = "complete"

    def _explain(self, job, features=None):
        try:
            bundle = self.bundle_provider()
            folder = job["folder"] / f"revision_{job['revision']}"
            folder.mkdir(exist_ok=True)
            if self.engine is None:
                if self.explanation_factory:
                    self.engine = self.explanation_factory()
                else:
                    from integritree.ml.explainability import ExplanationEngine
                    self.engine = ExplanationEngine(self.settings.research_model_dir, self.settings.research_prepared_dir,
                                                   self.experiment.shap, self.folder / "background", bundle=bundle)
            if features is None:
                import pandas as pd
                from integritree.ml.features import FEATURE_COLUMNS
                features = pd.DataFrame([job["result"]["model_inputs"]], columns=FEATURE_COLUMNS)
            self.engine.cache_root = folder / "shap"
            values = self.engine.explain(features, [job["id"]], f"{job['id']}:{job['revision']}")
            from integritree.ml.explainability import waterfall, summarize
            models = {}
            for value in values:
                item = dict(value)
                item.pop("waterfall_path", None)
                # Preserve SHAP numbers; present the confirmed receipt clock convention.
                for feature in item["features"]:
                    if feature["feature"] == "hour_of_day":
                        feature["readable_value"] = f"Manila hour = {job['result']['derived']['hour_of_day']:g}"
                    elif feature["feature"] == "day_of_week":
                        feature["readable_value"] = f"Manila weekday (Monday=0) = {job['result']['derived']['day_of_week']:g}"
                top, narrative = summarize([f["contribution"] for f in item["features"]],
                                           [f["readable_value"] for f in item["features"]], self.experiment.shap.positive_tolerance)
                item.update(top_positive_contributor=top, narrative=narrative, chart_description=narrative)
                model = item["model"]
                waterfall(item, folder / f"{model}.svg", display_label="Receipt")
                item["waterfall_url"] = f"/api/v1/receipts/{job['id']}/waterfall/{model}?revision={job['revision']}"
                models[model] = item
            if set(models) != {"rf", "rf_smote"}:
                raise ValueError("Both explanations required")
            with self.lock:
                job["result"]["explanation"] = {"status": "computed", "models": models}
        except Exception:
            with self.lock:
                job["result"]["explanation"] = {"status": "failed", "error": "Explanation unavailable. Retry is available."}

    def retry(self, token, identifier):
        with self.lock:
            job = self.owned(token, identifier)
            if job["status"] == "failed" and job["source"] is None:
                self._schedule(job, self._ocr)
                job["status"] = "queued"
            elif job["status"] == "failed" and job["confirmation"]:
                self._schedule(job, self._predict)
                job["status"] = "predicting"
            elif job["status"] == "complete" and job["result"]["explanation"]["status"] == "failed":
                self._schedule(job, self._explain)
                job["result"]["explanation"] = {"status": "pending"}
            else:
                raise ReceiptError("Nothing to retry for this receipt.", 409)
            return {"status": "pending"}

    def current(self, token, identifier, revision):
        job = self.owned(token, identifier)
        if job["busy"] or job["status"] != "complete" or job["revision"] != revision:
            raise ReceiptError("Result is unavailable or the receipt revision has changed.", 409)
        return job

    def image(self, token, identifier):
        with self.lock:
            job = self.owned(token, identifier)
            if "media_type" not in job:
                raise ReceiptError("Image preview is not ready.", 409)
            return (job["folder"] / "image").read_bytes(), job["media_type"]

    def chart(self, token, identifier, model, revision):
        with self.lock:
            job = self.current(token, identifier, revision)
            path = job["folder"] / f"revision_{revision}" / f"{model}.svg"
            if job["result"]["explanation"]["status"] != "computed" or not path.exists():
                raise ReceiptError("Waterfall is unavailable.", 409)
            return path.read_bytes()

    def export(self, token, identifier, revision):
        with self.lock:
            job = self.current(token, identifier, revision)
            bundle = self.bundle_provider()
            result = job["result"]
            metadata = {"scope": "experimental_receipt", "analysis_id": identifier, "revision": revision,
                        "input_sha256": result["input_sha256"], "mapping_version": result["mapping_version"],
                        "model_run_id": result["model_run_id"], "threshold": result["threshold"],
                        "preprocessing": bundle.preprocessor.state.model_dump(),
                        "model_package_versions": bundle.metadata.get("package_versions"),
                        "explanation_status": result["explanation"]["status"],
                        "retention": "temporary until Clear or backend shutdown/restart"}
            model_meta = self.settings.research_model_dir / "metadata.json"
            if model_meta.exists():
                metadata["model_metadata_sha256"] = file_sha256(model_meta)
            out = io.BytesIO()
            with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                files = {"confirmed_inputs.json": job["confirmation"].model_dump(mode="json"),
                         "results.json": {k: v for k, v in result.items() if k != "explanation"},
                         "explanations.json": result["explanation"], "metadata.json": metadata}
                for name, value in files.items():
                    archive.writestr(name, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
                if result["explanation"]["status"] == "computed":
                    for model in ("rf", "rf_smote"):
                        archive.write(job["folder"] / f"revision_{revision}" / f"{model}.svg", f"{model}_waterfall.svg")
            return out.getvalue()

    def delete(self, token, identifier):
        with self.lock:
            job = self.owned(token, identifier)
            if job["busy"]:
                raise ReceiptError("Wait for processing to finish before clearing the receipt.", 409)
            self._remove(job["folder"])
            del self.jobs[identifier]
