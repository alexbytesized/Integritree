"""Research commands validate configuration and expose supported entry points."""

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
        + (
            ["--prepared", "data/prepared/synthetic"]
            if script == "train_models"
            else []
        )
        + (
            ["--prepared", "data/prepared/synthetic", "--models", "artifacts/synthetic"]
            if script == "evaluate_models"
            else []
        ),
        env={**os.environ, "INTEGRITREE_BACKEND_ROOT": str(settings.backend_root)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "unresolved:" in result.stderr
    assert "preprocessing.features" in result.stderr


@pytest.mark.parametrize(
    "command", ["integritree-select", "integritree-select-three-stage"]
)
def test_selection_entry_points_use_three_stage(command):
    from importlib.metadata import distribution

    entries = {
        entry.name: entry.value
        for entry in distribution("integritree-backend").entry_points
    }
    assert entries[command] == "integritree.ml.staged_selection:main"
    assert "integritree-compare-ratios" not in entries
    result = subprocess.run(
        [str(Path(sys.executable).with_name(command + ".exe")), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--protocol" in result.stdout and "--stop-after-stage" in result.stdout
    assert "--baseline" not in result.stdout
