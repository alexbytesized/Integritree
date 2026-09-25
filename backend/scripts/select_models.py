"""Execute the predefined validation search; never evaluate test records."""
from integritree.ml.selection import main
if __name__ == "__main__":
    raise SystemExit(main())
