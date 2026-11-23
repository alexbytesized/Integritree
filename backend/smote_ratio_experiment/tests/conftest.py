import os
from pathlib import Path
import sys
import uuid

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
os.environ["MPLCONFIGDIR"] = str(ROOT / "runtime" / "matplotlib")
os.environ["MPLBACKEND"] = "Agg"


def pytest_configure(config):
    if config.option.basetemp is None:
        (ROOT / "runtime" / "pytest").mkdir(parents=True, exist_ok=True)
        config.option.basetemp = str(ROOT / "runtime" / "pytest" / uuid.uuid4().hex)
