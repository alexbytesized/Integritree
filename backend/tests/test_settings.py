import pytest
from pydantic import ValidationError

from integritree.settings import load_settings


def test_relative_paths_are_independent_of_shell_directory(tmp_path, monkeypatch):
    root = tmp_path / "backend"
    root.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    settings = load_settings(root)
    assert settings.dataset_path == root / "data/raw/ps_raw.csv"
    assert settings.experiment_config == root / "configs/experiment.yaml"
    assert not settings.uploads_dir.exists()


def test_environment_overrides_root_local_dotenv(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text(
        "INTEGRITREE_DATASET_PATH=data/from_file.csv\n", encoding="utf-8"
    )
    assert load_settings(tmp_path).dataset_path == tmp_path / "data/from_file.csv"
    monkeypatch.setenv("INTEGRITREE_DATASET_PATH", "data/from_environment.csv")
    assert load_settings(tmp_path).dataset_path == tmp_path / "data/from_environment.csv"


def test_root_environment_and_absolute_dataset_path(tmp_path, monkeypatch):
    root = tmp_path / "backend"
    root.mkdir()
    external = tmp_path / "source" / "data.csv"
    monkeypatch.setenv("INTEGRITREE_BACKEND_ROOT", str(root))
    monkeypatch.setenv("INTEGRITREE_DATASET_PATH", str(external))
    assert load_settings().backend_root == root
    assert load_settings().dataset_path == external


def test_relative_root_rejected(monkeypatch):
    monkeypatch.setenv("INTEGRITREE_BACKEND_ROOT", "relative")
    with pytest.raises(ValueError, match="absolute"):
        load_settings()


def test_unknown_dotenv_setting_rejected(tmp_path):
    (tmp_path / ".env").write_text("INTEGRITREE_TYPO=wrong\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="extra_forbidden"):
        load_settings(tmp_path)


@pytest.mark.parametrize("origin", ["*", "http://localhost:5173/path", "file:///tmp", "https://user:pass@host"])
def test_invalid_cors_origin_rejected(tmp_path, monkeypatch, origin):
    import json
    monkeypatch.setenv("INTEGRITREE_CORS_ORIGINS", json.dumps([origin]))
    with pytest.raises(ValidationError, match="cors_origins"):
        load_settings(tmp_path)
