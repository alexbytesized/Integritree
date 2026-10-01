"""Opt-in real HTTP/WebSocket browser fixture; synthetic OCR/models only.

Run from the repo root:
backend/.venv/Scripts/python -m uvicorn receipt_browser_app:create_browser_app --app-dir backend/tests --factory --port 8001
"""

from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

from integritree.api.main import create_app
from integritree.settings import BACKEND_ROOT, Settings
from test_receipt_api import extraction, Explainer
from test_research_api import bundle, experiment


def create_browser_app():
    temporary = TemporaryDirectory(prefix="receipt_browser_")
    settings = Settings(
        backend_root=Path(temporary.name).resolve(),
        experiment_config=BACKEND_ROOT / "configs/experiment.yaml",
        cors_origins=["http://127.0.0.1:5174"],
    )
    app = create_app(settings)
    original_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(application):
        try:
            async with original_lifespan(application):
                application.state.research.bundle = bundle.__wrapped__(
                    experiment.__wrapped__()
                )
                application.state.receipts.extractor = lambda _: extraction()
                application.state.receipts.explanation_factory = Explainer
                yield
        finally:
            temporary.cleanup()

    app.router.lifespan_context = lifespan
    return app
