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
                "rf": {"features": features},
                "rf_smote": {"features": deepcopy(features)},
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


def test_two_landscape_pages_with_correct_models_and_complete_text():
    result = sample_result()
    original = deepcopy(result)
    reader = read_pdf(result)
    assert result == original
    assert len(reader.pages) == 2
    for page, name, score, prediction, band, model in zip(
        reader.pages,
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
    assert "87.25 out of 100" in reader.pages[1].extract_text()


def test_long_explanation_fits_and_overflow_is_explicit():
    result = sample_result()
    result["original"]["reference"] = "0123456789ABCDEF" * 16
    for item in result["explanation"]["models"].values():
        item["features"][0]["readable_value"] = "Long transaction detail " * 8
    assert len(read_pdf(result).pages) == 2
    result["explanation"]["models"]["rf"]["features"][0]["readable_value"] *= 100
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
