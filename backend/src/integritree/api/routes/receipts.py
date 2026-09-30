"""Local session-owned receipt HTTP boundary; source context is never client-supplied."""
import hashlib
from typing import Annotated, Literal

from fastapi import APIRouter, Header, Query, Request
from fastapi.responses import Response
from pydantic import Field, StrictBool, ValidationError
from starlette.concurrency import run_in_threadpool

from integritree.receipts.contracts import ReceiptContract, ConfirmedReceiptFields
from integritree.services.receipts import ReceiptError, MAX_BYTES

router = APIRouter(prefix="/api/v1/receipts", tags=["receipts"])
Session = Annotated[str | None, Header(alias="X-Receipt-Session")]


class ConfirmationRequest(ReceiptContract):
    fields: ConfirmedReceiptFields
    confirmed: StrictBool
    expected_revision: Annotated[int, Field(strict=True, ge=0)]


@router.post("/sessions", status_code=201)
def session(request: Request):
    return {"token": request.app.state.receipts.session()}


@router.post("", status_code=202)
async def upload(request: Request, filename: str = Query(min_length=1, max_length=255), x_session: Session = None):
    service = request.app.state.receipts
    service.check_session(x_session)
    if request.headers.get("content-type", "").split(";")[0] not in ("image/png", "image/jpeg"):
        raise ReceiptError("Send one PNG/JPEG as the request body.", 415)
    length = request.headers.get("content-length")
    if length and (not length.isdigit() or int(length) > MAX_BYTES):
        raise ReceiptError("Image exceeds 10 MiB.", 413)
    job = service.reserve(x_session, filename)
    digest = hashlib.sha256()
    try:
        with (job["folder"] / "image").open("wb") as stream:
            async for chunk in request.stream():
                job["bytes_received"] += len(chunk)
                if job["bytes_received"] > MAX_BYTES:
                    raise ReceiptError("Image exceeds 10 MiB.", 413)
                digest.update(chunk)
                await run_in_threadpool(stream.write, chunk)
        if not job["bytes_received"]:
            raise ReceiptError("The image is empty.")
        service.submit(job, digest.hexdigest())
    except BaseException:
        service.abort(job)
        raise
    return {"id": job["id"], "status": job["status"]}


@router.get("/{identifier}")
def status(request: Request, identifier: str, x_session: Session = None):
    return request.app.state.receipts.status(x_session, identifier)


@router.post("/{identifier}/confirm", status_code=202)
async def confirm(request: Request, identifier: str, x_session: Session = None):
    service = request.app.state.receipts
    service.owned(x_session, identifier)
    # Bound input before JSON parsing; sanitize errors rather than echoing personal fields.
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > 16_384:
            raise ReceiptError("Confirmation is too large.", 413)
    try:
        value = ConfirmationRequest.model_validate_json(body)
    except ValidationError as exc:
        raise ReceiptError("Check the confirmation fields.", 422,
                           exc.errors(include_input=False, include_context=False, include_url=False)) from None
    return await run_in_threadpool(service.confirm, x_session, identifier, value.fields,
                                   value.confirmed, value.expected_revision)


@router.post("/{identifier}/retry", status_code=202)
def retry(request: Request, identifier: str, x_session: Session = None):
    return request.app.state.receipts.retry(x_session, identifier)


@router.get("/{identifier}/image")
def image(request: Request, identifier: str, x_session: Session = None):
    content, media = request.app.state.receipts.image(x_session, identifier)
    return Response(content, media_type=media)


@router.get("/{identifier}/waterfall/{model}")
def waterfall(request: Request, identifier: str, model: Literal["rf", "rf_smote"],
              revision: int = Query(ge=1), x_session: Session = None):
    return Response(request.app.state.receipts.chart(x_session, identifier, model, revision), media_type="image/svg+xml")


@router.get("/{identifier}/download")
def download(request: Request, identifier: str, revision: int = Query(ge=1), x_session: Session = None):
    return Response(request.app.state.receipts.export(x_session, identifier, revision), media_type="application/zip",
                    headers={"Content-Disposition": 'attachment; filename="integritree_receipt.zip"'})


@router.delete("/{identifier}", status_code=204)
def delete(request: Request, identifier: str, x_session: Session = None):
    request.app.state.receipts.delete(x_session, identifier)
    return Response(status_code=204)
