"""Keep generated synthetic test evidence inside the experiment."""

from pathlib import Path


def pytest_configure(config):
    if config.option.basetemp is None:
        config.option.basetemp = str(
            Path(__file__).resolve().parents[1] / "runtime" / "pytest"
        )
    Path(config.option.basetemp).resolve().parent.mkdir(parents=True, exist_ok=True)
