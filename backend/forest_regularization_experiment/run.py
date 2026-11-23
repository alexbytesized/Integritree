"""Isolated forest experiment entry point; never changes application settings."""

import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "src"))

from experiment import main

if __name__ == "__main__":
    raise SystemExit(main(ROOT))
