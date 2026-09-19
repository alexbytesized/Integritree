from fastapi.testclient import TestClient
import pytest

from integritree import __version__
from integritree.api.main import create_app


def test_health_without_dataset_or_models(settings):
    assert not settings.dataset_path.exists()
    assert not settings.artifacts_dir.exists()
    with TestClient(create_app(settings)) as client:
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        assert response.json() == {
            "status": "ok", "service": "integritree-backend", "version": __version__,
        }
        assert str(settings.backend_root) not in response.text
        schema = client.get("/openapi.json").json()
        assert set(schema["paths"]) == {"/api/v1/health"}
        assert client.get("/docs").status_code == 200
        assert client.post("/api/v1/predictions", json={}).status_code == 404
    assert not settings.artifacts_dir.exists()
    assert not settings.uploads_dir.exists()


def test_cors_for_configured_frontend_only(settings):
    with TestClient(create_app(settings)) as client:
        good = client.get("/api/v1/health", headers={"Origin": "http://localhost:5173"})
        assert good.headers["access-control-allow-origin"] == "http://localhost:5173"
        bad = client.get("/api/v1/health", headers={"Origin": "https://unlisted.example"})
        assert "access-control-allow-origin" not in bad.headers


def test_bad_config_prevents_startup(settings):
    settings.experiment_config.write_text("invalid: [", encoding="utf-8")
    with pytest.raises(ValueError, match="configuration"):
        create_app(settings)
