"""Run the isolated ordinary-SMOTE ratio experiment in the backend environment."""

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT.parent))
os.environ["MPLCONFIGDIR"] = str(ROOT / "runtime" / "matplotlib")
os.environ["MPLBACKEND"] = "Agg"

from ratio_experiment.runner import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(ROOT))
