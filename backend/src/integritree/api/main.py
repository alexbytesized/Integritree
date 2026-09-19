"""Phase 1 application factory: health only, with no model or dataset loading."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from integritree import __version__
from integritree.api.schemas import HealthResponse
from integritree.config import load_experiment
from integritree.settings import Settings, load_settings


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()
    experiment = load_experiment(settings.experiment_config)
    app = FastAPI(title="Integritree Backend", version=__version__)
    app.state.settings = settings
    app.state.experiment = experiment
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET"],
        allow_headers=[],
    )

    @app.get("/api/v1/health", response_model=HealthResponse, tags=["health"])
    def health() -> HealthResponse:
        """Check the application, not model readiness or dataset availability."""
        return HealthResponse(version=__version__)

    return app
