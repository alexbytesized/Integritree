from pathlib import Path
from uuid import uuid4


def pytest_configure(config):
    runtime = Path(__file__).resolve().parents[1] / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    if not config.option.basetemp:
        config.option.basetemp = str(runtime / ("pytest_" + uuid4().hex))
