"""One local OCR job in the isolated OCR environment. No network or console text."""

import json
import os
from pathlib import Path
import socket
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("OMP_THREAD_LIMIT", "2")


def offline(*args, **kwargs):
    raise RuntimeError("Receipt OCR is offline")


def main():
    socket.socket.connect = offline
    socket.socket.connect_ex = offline
    socket.create_connection = offline
    from integritree.receipts.ocr import RapidEngine, extract
    from integritree.receipts.gcash import parse_candidates
    from importlib.metadata import version

    engine = RapidEngine()
    try:
        value = extract(engine, Path(sys.argv[1]))
        value["parsed"] = parse_candidates(value["lines"])
        value["extractor_version"] = "rapidocr_" + version("rapidocr")
        with Path(sys.argv[2]).open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False)
    finally:
        engine.close()


if __name__ == "__main__":
    main()
