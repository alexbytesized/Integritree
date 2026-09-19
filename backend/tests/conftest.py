"""Synthetic foundation fixtures; never read the real PaySim dataset."""

import os

import pytest
import yaml

from integritree.settings import Settings


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    for name in list(os.environ):
        if name.startswith("INTEGRITREE_"):
            monkeypatch.delenv(name)


@pytest.fixture
def draft():
    return {
        "experiment_name": "synthetic_foundation_test",
        "dataset": {
            "filename": "synthetic.csv",
            "source_url": "https://example.com/synthetic",
            "sha256": "a" * 64,
            "row_count": 10,
        },
    }


@pytest.fixture
def settings(tmp_path, draft):
    root = tmp_path / "backend"
    config_dir = root / "configs"
    config_dir.mkdir(parents=True)
    (config_dir / "experiment.yaml").write_text(yaml.safe_dump(draft), encoding="utf-8")
    return Settings(backend_root=root)


@pytest.fixture
def raw_record():
    return {
        "step": 1, "type": "TRANSFER", "amount": 100.0,
        "nameOrig": "C_TEST_SENDER", "nameDest": "C_TEST_RECIPIENT",
        "oldbalanceOrg": 150.0, "newbalanceOrig": 50.0,
        "oldbalanceDest": 20.0, "newbalanceDest": 120.0,
        "isFraud": 1, "isFlaggedFraud": 0,
    }
