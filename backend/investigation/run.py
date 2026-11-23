"""Run validation diagnostics using the existing backend environment."""

import os
from pathlib import Path
import sys

# Set before importing backend plotting/SHAP modules; contain all generated caches.
ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
os.environ["MPLCONFIGDIR"] = str(ROOT / "runtime" / "matplotlib")
os.environ["MPLBACKEND"] = "Agg"

from diagnostics.runner import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(ROOT))
