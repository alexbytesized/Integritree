"""Fail clearly for research commands whose implementation belongs to later phases."""

import argparse
import sys
from pathlib import Path

from integritree.config import Stage, load_experiment
from integritree.settings import load_settings


def pending_command(description: str, phase: int, stage: Stage | None = None) -> int:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, help="Path relative to the backend root")
    args = parser.parse_args()
    try:
        settings = load_settings()
        path = args.config or settings.experiment_config
        if not path.is_absolute():
            path = settings.backend_root / path
        config = load_experiment(path)
        if stage is not None:
            config.require_ready(stage)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(f"{description} is not implemented yet (Phase {phase}).", file=sys.stderr)
    return 2
