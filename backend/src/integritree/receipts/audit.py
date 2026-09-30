"""Local receipt inventory. Does not read PaySim or load fraud models."""
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
import warnings

from PIL import Image, ImageOps

MAX_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 20_000_000
CATEGORIES = ("TRANSFER", "CASH_IN", "CASH_OUT", "PAYMENT", "DEBIT", "UNSURE")
# Already inspected in planning: these cannot be unseen verification examples.
PREVIEWED = {
    "TRANSFER/transfer_0001.jpg", "PAYMENT/payment_0001.jpg",
    "PAYMENT/payment_0007.jpg", "PAYMENT/payment_0019.jpg",
    "DEBIT/debit_0001.jpg", "DEBIT/debit_0004.jpg",
}


def file_hash(path):
    with Path(path).open("rb") as stream:
        digest = sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_image(path):
    """Validate bytes and decoded dimensions before allocating an RGB image."""
    path = Path(path)
    if path.stat().st_size > MAX_BYTES:
        raise ValueError("image_exceeds_10_MiB")
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(path) as image:
            if image.format not in {"PNG", "JPEG"}:
                raise ValueError("unsupported_image_format")
            if image.width * image.height > MAX_PIXELS:
                raise ValueError("image_exceeds_20_megapixels")
            if getattr(image, "n_frames", 1) != 1:
                raise ValueError("animated_image_not_supported")
            image.load()
            return ImageOps.exif_transpose(image).convert("RGB")


def inventory(root):
    root = Path(root).resolve()
    rows = []
    for category in CATEGORIES:
        for path in sorted((root / category).glob("*")):
            if not path.is_file():
                continue
            if path.is_symlink() or not path.resolve().is_relative_to(root):
                raise ValueError("sample_path_outside_collection")
            row = {"path": path.relative_to(root).as_posix(), "category": category,
                   "bytes": path.stat().st_size, "sha256": file_hash(path)}
            try:
                with load_image(path) as image:
                    row.update(width=image.width, height=image.height,
                               pixel_sha256=sha256(
                                   str(image.size).encode() + image.tobytes()).hexdigest(),
                               status="valid")
            except (ValueError, OSError, Image.DecompressionBombError,
                    Image.DecompressionBombWarning) as exc:
                row.update(status="invalid", error=type(exc).__name__)
            rows.append(row)
    groups = defaultdict(list)
    for row in rows:
        if row["status"] == "valid":
            groups[row["pixel_sha256"]].append(row["path"])
    return {"schema_version": 1, "rows": rows,
            "counts": {c: sum(r["category"] == c for r in rows) for c in CATEGORIES},
            "status_counts": dict(Counter(r["status"] for r in rows)),
            "decoded_duplicate_groups": [g for g in groups.values() if len(g) > 1],
            "limits": {"bytes": MAX_BYTES, "decoded_pixels": MAX_PIXELS},
            "note": "Pixel-identical duplicates grouped; near duplicates require review."}


def split_inventory(audit):
    """Group pixel-identical copies; reserve deterministic ~20%, minimum 2/type.

    Previously viewed screenshots (and their duplicates) stay in development.
    This is an OCR split, entirely separate from the PaySim held-out experiment.
    """
    groups = defaultdict(list)
    for row in audit["rows"]:
        if row["status"] == "valid":
            groups[row["pixel_sha256"]].append(row)
    assigned = {}
    for category in CATEGORIES:
        candidates = [key for key, rows in groups.items()
                      if {r["category"] for r in rows} == {category}]
        candidates.sort(key=lambda key: sha256(("receipt-42:" + key).encode()).hexdigest())
        eligible = [key for key in candidates
                    if not any(r["path"] in PREVIEWED for r in groups[key])]
        count = min(max(2, round(len(candidates) * .2)), max(0, len(candidates) - 2), len(eligible))
        held = set(eligible[:count]) if category != "UNSURE" else set()
        for key in candidates:
            assigned[key] = "verification" if key in held else "development"
    result = []
    for row in audit["rows"]:
        row = dict(row)
        row["split"] = assigned.get(row.get("pixel_sha256"), "review")
        result.append(row)
    return result
