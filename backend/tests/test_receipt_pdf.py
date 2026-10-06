"""PDF content/layout contracts and parity with the screen's display helpers."""

import io
import json
import shutil
import subprocess
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import pytest
from pypdf import PdfReader

from integritree.services import receipt_pdf


def sample_result():
    features = [
        {
            "feature": "type_TRANSFER",
            "readable_value": "Transaction type is TRANSFER",
            "contribution": 0.0483,
        },
        {"feature": "day_of_week", "contribution": 0.008},
        {"feature": "hour_of_day", "contribution": -0.0063},
        {
            "feature": "log_amount",
            "readable_value": "Transaction amount",
            "contribution": -0.002,
        },
        {"feature": "ignored", "contribution": 1e-10},
    ]
    return {
        "original": {"reference": "8029858977412"},
        "derived": {"day_of_week": 4, "hour_of_day": 12},
        "threshold": 0.43,
        "rf": {"score": 0.0107, "predicted_label": 0},
        "rf_smote": {"score": 0.8725, "predicted_label": 1},
        "explanation": {
            "status": "computed",
            "models": {
                model: {
                    "features": [
                        {
                            "feature": name,
                            "readable_value": name,
                            "contribution": next(
                                (
                                    f["contribution"]
                                    for f in features
                                    if f["feature"] == name
                                ),
                                0,
                            )
                            * factor,
                        }
                        for name in receipt_pdf.FEATURE_COLUMNS
                    ],
                    "base_value": score - 0.048 * factor,
                    "output_value": score,
                    "top_positive_contributor": {
                        "status": "available",
                        "feature": "type_TRANSFER",
                    },
                }
                for model, score, factor in (
                    ("rf", 0.0107, 0.1),
                    ("rf_smote", 0.8725, 1),
                )
            },
        },
    }


def read_pdf(result):
    return PdfReader(io.BytesIO(receipt_pdf.generate_pdf(result)))


@pytest.mark.parametrize(
    "instant,date",
    [
        ("2026-10-05T15:59:59+00:00", "2026-10-05"),
        ("2026-10-05T16:00:00+00:00", "2026-10-06"),
        ("2026-12-31T16:00:00+00:00", "2027-01-01"),
    ],
)
def test_philippine_filename(instant, date):
    assert (
        receipt_pdf.pdf_filename(datetime.fromisoformat(instant))
        == f"Transaction-Results_{date}.pdf"
    )


def test_four_landscape_pages_with_correct_models_and_complete_text():
    result = sample_result()
    original = deepcopy(result)
    reader = read_pdf(result)
    assert result == original
    assert len(reader.pages) == 4
    for page, name, score, prediction, band, model in zip(
        reader.pages[::2],
        ("Benchmark RF", "RF-SMOTE"),
        ("1.07", "87.25"),
        ("Legitimate", "Fraudulent"),
        ("Minimal Risk", "Critical Risk"),
        ("rf", "rf_smote"),
    ):
        assert float(page.mediabox.width) == pytest.approx(841.89, abs=0.01)
        assert float(page.mediabox.height) == pytest.approx(595.28, abs=0.01)
        text = " ".join(page.extract_text().split())
        assert f"Transaction Record · {name}" in text
        assert "Transaction ID: 8029858977412" in text
        assert f"{score} out of 100" in text
        assert (
            f"The {name} model classified the transaction as {prediction}, with a fraud risk score of {score}%, corresponding to a {band} level."
            in text
        )
        assert "Fraud threshold: 43.00%" in text
        assert (
            receipt_pdf.shap_summary(result["explanation"], model, result["derived"])
            in text
        )
        for forbidden in (
            "Ground Truth",
            "Outcome",
            "See Full SHAP",
            "Download Results",
        ):
            assert forbidden not in text
        fonts = [font.get_object() for font in page["/Resources"]["/Font"].values()]
        assert any(
            "DejaVuSans" in str(font) and "/FontDescriptor" in font for font in fonts
        )


@pytest.mark.parametrize(
    "reference", [None, "  ", "0123456789ABCDEF" * 16, "<b>&Reference</b>"]
)
def test_missing_long_and_markup_reference(reference):
    result = sample_result()
    result["original"]["reference"] = reference
    reader = read_pdf(result)
    expected = reference.strip() if reference else ""
    for page in reader.pages:
        text = page.extract_text()
        assert (expected or "Not provided").replace(" ", "") in "".join(text.split())


def test_failed_shap_keeps_both_predictions():
    result = sample_result()
    result["explanation"] = {"status": "failed"}
    reader = read_pdf(result)
    for page in reader.pages:
        text = " ".join(page.extract_text().split())
        assert (
            "The SHAP explanation is unavailable. Retry it in the application" in text
        )
    assert "1.07 out of 100" in reader.pages[0].extract_text()
    assert "87.25 out of 100" in reader.pages[2].extract_text()


def test_long_explanation_fits_and_overflow_is_explicit():
    result = sample_result()
    result["original"]["reference"] = "0123456789ABCDEF" * 16
    for item in result["explanation"]["models"].values():
        next(f for f in item["features"] if f["feature"] == "type_TRANSFER")[
            "readable_value"
        ] = "Long transaction detail " * 8
    assert len(read_pdf(result).pages) == 4
    next(
        f
        for f in result["explanation"]["models"]["rf"]["features"]
        if f["feature"] == "type_TRANSFER"
    )["readable_value"] *= 100
    with pytest.raises(ValueError, match="too long"):
        receipt_pdf.generate_pdf(result)


def test_display_parity_with_frontend_using_shared_synthetic_cases():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required for the frontend/backend display parity check")
    display = Path(__file__).resolve().parents[2] / "frontend/src/researchDisplay.js"
    scores = [0, 1.005, 1.125, 19.999, 20, 39.999, 40, 59.999, 60, 79.999, 80, 100]
    cases = [sample_result()]
    floating_derived = sample_result()
    floating_derived["derived"] = {"hour_of_day": 13.0, "day_of_week": 0.0}
    cases.append(floating_derived)
    for features in (
        [],
        [{"feature": "zero", "contribution": 0}],
        [
            {"feature": "type_CASH_IN", "contribution": 0.02},
            {"feature": "type_PAYMENT", "contribution": 0.01},
            {"feature": "type_DEBIT", "contribution": 0.001},
        ],
    ):
        item = sample_result()
        item["explanation"]["models"]["rf"]["features"] = features
        cases.append(item)
    script = """
        import { readFileSync } from 'node:fs';
        const { riskBand, scoreText, shapSummary } = await import(process.argv[1]);
        const { scores, cases } = JSON.parse(readFileSync(0, 'utf8'));
        console.log(JSON.stringify({scores: scores.map(s => [scoreText(s), riskBand(s).label]),
            summaries: cases.map(c => shapSummary(c.explanation, 'rf', c.derived, 'receipt'))}));
    """
    completed = subprocess.run(
        [node, "--input-type=module", "-e", script, display.as_uri()],
        input=json.dumps({"scores": scores, "cases": cases}),
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=True,
        timeout=30,
    )
    actual = json.loads(completed.stdout)
    assert actual["scores"] == [
        [receipt_pdf.score_text(s), receipt_pdf.risk_band(s)[1]] for s in scores
    ]
    assert actual["summaries"] == [
        receipt_pdf.shap_summary(c["explanation"], "rf", c["derived"]) for c in cases
    ]


def test_gauge_markers_use_unrounded_positions_and_bounds():
    from unittest.mock import Mock

    canvas = Mock()
    receipt_pdf._gauge(canvas, 7.251, 43, 50, 500, 700)
    assert canvas.circle.call_args.args[:3] == pytest.approx((100.757, 492, 9))
    assert canvas.line.call_args.args == pytest.approx((351, 479, 351, 505))
    for score, expected_x in ((0, 50), (100, 750)):
        receipt_pdf._gauge(canvas, score, 43, 50, 500, 700)
        assert canvas.circle.call_args.args[0] == expected_x


@pytest.mark.parametrize(
    "values,expected",
    [
        ([0] * 11, (-40, 40, 10)),
        ([12, -8], (-40, 40, 10)),
        ([50], (-30, 50, 10)),
        ([-50], (-50, 30, 10)),
        ([-50, 50], (-80, 80, 20)),
        ([90], (-60, 100, 20)),
        ([-90], (-100, 60, 20)),
        ([100], (-60, 100, 20)),
        ([-100], (-100, 60, 20)),
        ([-100, 100], (-100, 100, 25)),
        ([1e-8, -1e-8], (-40, 40, 10)),
    ],
)
def test_pdf_grid_keeps_nine_lines_zero_and_all_effects(values, expected):
    ticks = receipt_pdf.contribution_ticks(values)
    assert (ticks[0], ticks[-1], ticks[1] - ticks[0]) == expected
    assert len(ticks) == 9 and 0 in ticks
    assert ticks[0] <= min([0, *values]) <= max([0, *values]) <= ticks[-1]


def test_pdf_evaluation_values_order_fonts_and_no_definitions():
    result = sample_result()
    for item in result["explanation"]["models"].values():
        item["features"].reverse()
    reader = read_pdf(result)
    for index, model in enumerate(("rf", "rf_smote")):
        page = reader.pages[index * 2 + 1]
        text = " ".join(page.extract_text().split())
        item = result["explanation"]["models"][model]
        assert item["base_value"] + sum(
            f["contribution"] for f in item["features"]
        ) == pytest.approx(item["output_value"])
        for title in (
            "Reference Score",
            "Output Score",
            "Top Risk-Increasing Contributor",
        ):
            assert title in text
        for key in ("base_value", "output_value"):
            assert f"{receipt_pdf.score_text(item[key] * 100)}%" in text
        assert "Contribution to Fraud Risk Score" not in text
        assert "SHAP Feature Contributions" not in text
        assert "starting score" not in text and "numerical tolerance" not in text
        assert "Risk Score" not in text.replace("Fraud Risk Score", "")
        offsets = [text.index(label) for label in receipt_pdf.FEATURE_LABELS.values()]
        assert offsets == sorted(offsets)
        assert not page.get("/Resources", {}).get("/XObject")  # Native vector content.
        fragments = []
        page.extract_text(
            visitor_text=lambda value, cm, tm, font, size, captured=fragments: (
                captured.append((value, font, size))
            )
        )
        for key in ("base_value", "output_value"):
            value = f"{receipt_pdf.score_text(item[key] * 100)}%"
            matches = [(font, size) for part, font, size in fragments if value in part]
            assert matches and all(
                "Bold" not in str(font["/BaseFont"]) and size == 11
                for font, size in matches
            )
    for index, page in enumerate(reader.pages):
        assert f"{index + 1} / 4" in page.extract_text()
        expected_model = "Benchmark RF" if index < 2 else "RF-SMOTE"
        assert expected_model in page.extract_text()


@pytest.mark.parametrize("status", ["pending", "failed"])
def test_unavailable_shap_retains_four_pages_and_summary_cards(status):
    result = sample_result()
    result["explanation"] = {"status": status}
    reader = read_pdf(result)
    assert len(reader.pages) == 4
    for page in reader.pages[1::2]:
        text = page.extract_text()
        assert "SHAP explanation is unavailable" in text
        assert text.count("Unavailable") == 3


def test_zero_and_missing_feature_data_are_distinct():
    result = sample_result()
    for item in result["explanation"]["models"].values():
        for feature in item["features"]:
            feature["contribution"] = 0
        item["output_value"] = item["base_value"]
        item["top_positive_contributor"] = {"status": "no_positive_contributor"}
    for page in read_pdf(result).pages[1::2]:
        text = page.extract_text()
        assert "No transaction details meaningfully increased the score." in text
        assert "0 pp" not in text
        assert "unavailable" not in text
    result["explanation"]["models"]["rf"]["features"].pop()
    assert "unavailable" in read_pdf(result).pages[1].extract_text()


def test_vector_bar_geometry_end_labels_and_tiny_markers(monkeypatch):
    from reportlab.lib import colors
    from reportlab.pdfgen.canvas import Canvas

    receipt_pdf._register_fonts()
    canvas = Canvas(io.BytesIO())
    bars, lines, labels = [], [], []
    original_bar, original_line = canvas.roundRect, canvas.line
    original_left, original_right = canvas.drawString, canvas.drawRightString

    def bar(x, y, width, height, radius, **kwargs):
        bars.append((x, width, canvas._fillColorObj))
        return original_bar(x, y, width, height, radius, **kwargs)

    def line(*args):
        lines.append(args)
        return original_line(*args)

    def left(x, y, text):
        labels.append(
            (x, x + receipt_pdf.pdfmetrics.stringWidth(text, "Receipt", 9), text)
        )
        return original_left(x, y, text)

    def right(x, y, text):
        labels.append(
            (x - receipt_pdf.pdfmetrics.stringWidth(text, "Receipt", 9), x, text)
        )
        return original_right(x, y, text)

    monkeypatch.setattr(canvas, "roundRect", bar)
    monkeypatch.setattr(canvas, "line", line)
    monkeypatch.setattr(canvas, "drawString", left)
    monkeypatch.setattr(canvas, "drawRightString", right)
    values = [1, -1, 0.4, -0.4, 1e-10, -1e-10] + [0] * 5
    features = [
        {"feature": name, "contribution": value}
        for name, value in zip(receipt_pdf.FEATURE_COLUMNS, values)
    ]
    receipt_pdf._contribution_graph(canvas, features, 36, 510, 770, 260)
    assert len(bars) == 6
    assert len(lines) == 11  # Nine gridlines and two tiny-effect markers.
    axis_left, axis_right = lines[0][0], lines[8][0]
    scale = (axis_right - axis_left) / 200
    zero = lines[4][0]
    for (x, width, color), value in zip(bars, values):
        assert width == pytest.approx(abs(value) * 100 * scale)
        assert x == pytest.approx(zero + min(0, value) * 100 * scale)
        assert color == colors.HexColor(
            receipt_pdf.INCREASE_COLOR if value > 0 else receipt_pdf.DECREASE_COLOR
        )
    contributions = [
        (left, right, text) for left, right, text in labels if text.endswith(" pp")
    ]
    assert [text for _, _, text in contributions] == [
        f"{value * 100:+.3g} pp" for value in values if value != 0
    ]
    name_right = max(
        right
        for _, right, text in labels
        if text in receipt_pdf.FEATURE_LABELS.values()
    )
    assert all(name_right < left < right <= 806 for left, right, _ in contributions)
