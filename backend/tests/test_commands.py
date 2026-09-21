"""Unimplemented commands must never report a successful research run."""

import os
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.mark.parametrize("script", ["prepare_data", "train_models", "evaluate_models"])
def test_research_commands_reject_unresolved_draft(script, settings):
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / f"{script}.py")]
        + (["--prepared", "data/prepared/synthetic"] if script == "train_models" else []),
        env={**os.environ, "INTEGRITREE_BACKEND_ROOT": str(settings.backend_root)},
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 2
    assert "unresolved:" in result.stderr
    assert "preprocessing.features" in result.stderr


def test_batch_command_reports_unimplemented(settings):
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / "predict_batch.py")],
        env={**os.environ, "INTEGRITREE_BACKEND_ROOT": str(settings.backend_root)},
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 2
    assert "not implemented yet (Phase 5)" in result.stderr
