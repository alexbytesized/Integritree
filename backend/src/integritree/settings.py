"""Local application settings, independent of experiment hyperparameters."""

import os
from pathlib import Path
from typing import Self

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="INTEGRITREE_", extra="forbid", env_file_encoding="utf-8"
    )

    backend_root: Path = BACKEND_ROOT
    dataset_path: Path = Path("data/raw/ps_raw.csv")
    experiment_config: Path = Path("configs/experiment.yaml")
    prepared_dir: Path = Path("data/prepared")
    artifacts_dir: Path = Path("artifacts")
    reports_dir: Path = Path("reports")
    uploads_dir: Path = Path("runtime/uploads")
    exports_dir: Path = Path("runtime/exports")
    cors_origins: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

    @field_validator("backend_root")
    @classmethod
    def absolute_root(cls, value: Path) -> Path:
        if not value.is_absolute():
            raise ValueError("backend_root must be an absolute directory")
        return value.resolve()

    @field_validator("cors_origins")
    @classmethod
    def explicit_origins(cls, values: list[str]) -> list[str]:
        from urllib.parse import urlsplit

        for value in values:
            parts = urlsplit(value)
            if (
                parts.scheme not in ("http", "https")
                or not parts.netloc
                or parts.hostname is None
                or "*" in value
                or parts.path
                or parts.query
                or parts.fragment
                or parts.username is not None
            ):
                raise ValueError("cors_origins must be explicit HTTP(S) origins without paths")
        return values

    @model_validator(mode="after")
    def resolve_paths(self) -> Self:
        for field in (
            "dataset_path", "experiment_config", "prepared_dir", "artifacts_dir",
            "reports_dir", "uploads_dir", "exports_dir",
        ):
            path = getattr(self, field)
            if not path.is_absolute():
                path = self.backend_root / path
            setattr(self, field, path.resolve())
        return self


def load_settings(backend_root: Path | None = None) -> Settings:
    """Use a root-local .env, with process environment overriding that file."""
    root = backend_root or Path(os.environ.get("INTEGRITREE_BACKEND_ROOT", BACKEND_ROOT))
    if not root.is_absolute():
        raise ValueError("INTEGRITREE_BACKEND_ROOT must be an absolute directory")
    root = root.resolve()
    return Settings(backend_root=root, _env_file=root / ".env")
