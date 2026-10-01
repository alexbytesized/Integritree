"""Synthetic receipt checks; no personal samples or OCR weights required."""

from PIL import Image
import pytest

from integritree.receipts import audit
from integritree.receipts.gcash import parse_candidates
from integritree.receipts.ocr import TextLine, ordered_lines


def parse(*texts):
    return parse_candidates([{"text": text} for text in texts])


def test_bank_principal_and_reference_are_separate_from_total_invoice_status_bar():
    result = parse(
        "08:59",
        "Bank Transfer",
        "Transfer Amount 1,234.50",
        "+Fee 15.00",
        "Total 1,249.50",
        "InstaPay Invoice No. 987654",
        "Transfer Date Jan 31, 2026 06:14 PM",
        "Ref No. 000123",
    )
    assert result["fields"] == {
        "workflow": "bank_transfer",
        "category": "DEBIT",
        "amount": "1234.50",
        "date": "2026-01-31",
        "time": "18:14",
        "reference": "000123",
    }
    assert result["requires_confirmation"] is True


def test_parser_does_not_fill_missing_principal_or_date_from_other_values():
    fields = parse("Pay Online", "Total 999.00", "+Fee 15.00", "12:34")["fields"]
    assert fields["amount"] is None
    assert fields["date"] is None
    assert fields["time"] is None
    assert fields["reference"] is None


def test_conflicting_candidates_remain_unresolved():
    result = parse(
        "Express Send",
        "Bank Transfer",
        "Amount 10.00",
        "Amount 11.00",
        "Ref No. 001",
        "Ref No. 002",
        "Jan 1, 2026 12:00 AM",
        "Jan 2, 2026 12:00 PM",
    )
    assert all(value is None for value in result["fields"].values())
    assert set(result["ambiguous_fields"]) == {
        "workflow",
        "amount",
        "reference",
        "datetime",
    }


@pytest.mark.parametrize("date", ["Feb 30, 2026 01:00 PM", "Jan 1, 2026 13:00 PM"])
def test_impossible_calendar_date_or_twelve_hour_time_is_rejected(date):
    assert parse(date)["fields"]["date"] is None


def test_split_lines_preserve_reference_zeros_and_decimal_principal():
    fields = parse(
        "Paid and linked via GCash",
        "Amount",
        "P 0.00",
        "Reference No.",
        "00001234",
        "Feb 1, 2026 12:00 AM",
    )["fields"]
    assert fields["workflow"] == "pay_online"
    assert fields["amount"] == "0.00"
    assert fields["reference"] == "00001234"
    assert fields["time"] == "00:00"


def test_payment_alone_does_not_prove_a_supported_workflow():
    assert parse("Paid via GCash", "Amount 12.00")["fields"]["workflow"] is None


def test_geometric_reading_order_joins_labels_and_values_without_joining_rows():
    lines = ordered_lines(
        [
            TextLine("35.00", 0.8, [100, 11, 130, 21]),
            TextLine("Total 50.00", 0.9, [0, 40, 130, 50]),
            TextLine("Amount", 0.95, [0, 10, 50, 20]),
        ]
    )
    assert [r.text for r in lines] == ["Amount 35.00", "Total 50.00"]
    assert lines[0].confidence == 0.8


def make_image(root, relative, color):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color).save(path)
    return path


def test_inventory_groups_duplicates_and_keeps_conflicts_out_of_both_splits(tmp_path):
    for n in range(12):
        make_image(tmp_path, f"TRANSFER/transfer_{n + 1:04d}.jpg", (n * 20, 0, 0))
    first = tmp_path / "TRANSFER/transfer_0001.jpg"
    (first.parent / "copy.jpg").write_bytes(first.read_bytes())
    conflict = make_image(tmp_path, "PAYMENT/conflict.png", (0, 255, 0))
    (first.parent / "conflict.png").write_bytes(conflict.read_bytes())
    report = audit.inventory(tmp_path)
    rows = audit.split_inventory(report)
    lookup = {r["path"]: r for r in rows}
    assert len(report["decoded_duplicate_groups"]) == 2
    assert lookup["TRANSFER/transfer_0001.jpg"]["split"] == "development"
    assert lookup["TRANSFER/copy.jpg"]["split"] == "development"
    assert lookup["TRANSFER/conflict.png"]["split"] == "review"
    assert lookup["PAYMENT/conflict.png"]["split"] == "review"
    dev = {r["pixel_sha256"] for r in rows if r["split"] == "development"}
    verification = {r["pixel_sha256"] for r in rows if r["split"] == "verification"}
    assert verification and dev.isdisjoint(verification)
    assert rows == audit.split_inventory(report)


def test_invalid_image_is_audited_without_raw_error_or_contents(tmp_path):
    path = tmp_path / "TRANSFER/bad.jpg"
    path.parent.mkdir()
    path.write_bytes(b"not an image")
    row = audit.split_inventory(audit.inventory(tmp_path))[0]
    assert row["status"] == "invalid" and row["split"] == "review"
    assert row["error"] == "UnidentifiedImageError"


def test_byte_and_pixel_limits(tmp_path, monkeypatch):
    path = make_image(tmp_path, "valid.png", "blue")
    monkeypatch.setattr(audit, "MAX_BYTES", path.stat().st_size - 1)
    with pytest.raises(ValueError, match="10_MiB"):
        audit.load_image(path)
    monkeypatch.setattr(audit, "MAX_BYTES", path.stat().st_size)
    monkeypatch.setattr(audit, "MAX_PIXELS", 63)
    with pytest.raises(ValueError, match="megapixels"):
        audit.load_image(path)


def test_image_content_is_checked_instead_of_extension(tmp_path):
    path = tmp_path / "fake.jpg"
    Image.new("RGB", (8, 8)).save(path, format="GIF")
    with pytest.raises(ValueError, match="unsupported_image_format"):
        audit.load_image(path)
