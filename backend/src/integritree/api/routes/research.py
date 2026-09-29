"""Session-owned researcher API. CSV bodies stream directly to temporary disk."""
import hashlib
from typing import Annotated, Literal
from fastapi import APIRouter, Header, Query, Request
from fastapi.responses import FileResponse, Response
from starlette.concurrency import run_in_threadpool
from integritree.services.research import MAX_BYTES, ResearchError

router = APIRouter(prefix="/api/v1/research", tags=["research"])
Session = Annotated[str | None, Header(alias="X-Research-Session")]


@router.post("/sessions", status_code=201)
def session(request: Request):
    return {"token": request.app.state.research.session()}


@router.get("/template")
def template():
    return Response("step,type,amount,nameOrig,nameDest,isFraud\n1,TRANSFER,100,C_DEMO_ORIGIN,C_DEMO_DESTINATION,0\n",
                    media_type="text/csv", headers={"Content-Disposition": 'attachment; filename="research_template.csv"'})


@router.post("/analyses", status_code=202)
async def upload(request: Request, x_session: Session = None, filename: str = Query(max_length=255)):
    service = request.app.state.research
    service.check_session(x_session)
    if request.headers.get("content-type", "").split(";")[0] not in ("text/csv", "application/octet-stream"):
        raise ResearchError("Send the CSV as the request body with Content-Type: text/csv.", 415)
    length = request.headers.get("content-length")
    if length and (not length.isdigit() or int(length) > MAX_BYTES):
        raise ResearchError("File exceeds the 500 MiB limit.", 413)
    job = service.reserve(x_session, filename)
    digest = hashlib.sha256()
    try:
        with (job["folder"] / "input.csv").open("wb") as stream:
            async for chunk in request.stream():
                job["bytes_received"] += len(chunk)
                if job["bytes_received"] > MAX_BYTES:
                    raise ResearchError("File exceeds the 500 MiB limit.", 413)
                digest.update(chunk)
                await run_in_threadpool(stream.write, chunk)
        if not job["bytes_received"]:
            raise ResearchError("The CSV file is empty.")
        service.submit(job, digest.hexdigest())
    except BaseException:
        service.abort_upload(job)
        raise
    return {"id": job["id"], "status": job["status"]}


@router.get("/analyses/{identifier}")
def status(request: Request, identifier: str, x_session: Session = None):
    return request.app.state.research.status(x_session, identifier)


@router.get("/analyses/{identifier}/records")
def records(request: Request, identifier: str, x_session: Session = None, page: int = Query(1, ge=1),
            search: str = Query("", max_length=200), model: Literal["both", "rf", "rf_smote"] = "both",
            outcome: Literal["all", "tp", "fp", "tn", "fn"] = "all",
            search_field: Literal["transaction_id", "row_number"] = "transaction_id"):
    return request.app.state.research.records(x_session, identifier, page, search, model, outcome, search_field)


@router.get("/analyses/{identifier}/records/{number}")
def record(request: Request, identifier: str, number: int, x_session: Session = None):
    return request.app.state.research.record(x_session, identifier, number)


@router.post("/analyses/{identifier}/records/{number}/explanation", status_code=202)
def explain(request: Request, identifier: str, number: int, x_session: Session = None):
    return request.app.state.research.explain(x_session, identifier, number)


@router.get("/analyses/{identifier}/records/{number}/waterfall/{model}")
def waterfall(request: Request, identifier: str, number: int, model: Literal["rf", "rf_smote"], x_session: Session = None,
              presentation: Literal["original", "row_number"] = "original"):
    service = request.app.state.research
    job = service.owned(x_session, identifier, True)
    item = service.record(x_session, identifier, number)
    path = job["folder"] / f"{number}_{model}.svg"
    if item["explanation"]["status"] != "computed" or not path.exists():
        raise ResearchError("Waterfall is not ready.", 409)
    if presentation == "row_number":
        path = service.display_waterfall(job, number, model, item["explanation"]["models"][model])
    return FileResponse(path, media_type="image/svg+xml", headers={"Cache-Control": "no-store"})


@router.post("/analyses/{identifier}/exports", status_code=202)
def export(request: Request, identifier: str, x_session: Session = None):
    return request.app.state.research.export(x_session, identifier)


@router.get("/analyses/{identifier}/exports/download")
def download(request: Request, identifier: str, x_session: Session = None):
    job = request.app.state.research.owned(x_session, identifier, True)
    if job["export"]["status"] != "complete":
        raise ResearchError("Export is not ready.", 409)
    return FileResponse(job["folder"] / "results.zip", filename="integritree_results.zip", media_type="application/zip")


@router.delete("/analyses/{identifier}", status_code=204)
def delete(request: Request, identifier: str, x_session: Session = None):
    request.app.state.research.delete(x_session, identifier)
    return Response(status_code=204)
