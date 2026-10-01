"""Local application; trusted models load lazily on the first research job."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from integritree import __version__
from integritree.api.schemas import HealthResponse
from integritree.config import load_experiment
from integritree.settings import Settings, load_settings
from integritree.api.routes.research import router
from integritree.services.research import ResearchService, ResearchError
from integritree.api.routes.receipts import router as receipt_router
from integritree.services.receipts import ReceiptService, ReceiptError


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()
    experiment = load_experiment(settings.experiment_config)

    @asynccontextmanager
    async def lifespan(app):
        app.state.research = ResearchService(settings, experiment)
        try:
            app.state.receipts = ReceiptService(
                settings, experiment, app.state.research.get_bundle
            )
            try:
                yield
            finally:
                app.state.receipts.close()
        finally:
            app.state.research.close()

    app = FastAPI(title="Integritree Backend", version=__version__, lifespan=lifespan)
    app.state.settings = settings
    app.state.experiment = experiment
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type", "X-Research-Session", "X-Receipt-Session"],
    )
    app.include_router(router)
    app.include_router(receipt_router)

    @app.middleware("http")
    async def receipt_privacy_headers(request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/api/v1/receipts"):
            response.headers["Cache-Control"] = "no-store"
            response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.exception_handler(ReceiptError)
    async def receipt_error(request, exc):
        return JSONResponse(
            status_code=exc.status, content={"detail": str(exc), "issues": exc.issues}
        )

    @app.exception_handler(ResearchError)
    async def research_error(request, exc):
        return JSONResponse(
            status_code=exc.status, content={"detail": str(exc), "issues": exc.issues}
        )

    @app.get("/api/v1/health", response_model=HealthResponse, tags=["health"])
    def health() -> HealthResponse:
        """Check the application, not model readiness or dataset availability."""
        return HealthResponse(version=__version__)

    return app
