"""Two-page receipt summaries from saved results, without inference or disk writes."""

import io
from datetime import datetime, timedelta, timezone
from decimal import ROUND_HALF_UP, Decimal
from functools import lru_cache
from html import escape
from pathlib import Path
from threading import Lock

import matplotlib
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Paragraph

MODELS = (("rf", "Benchmark RF"), ("rf_smote", "RF-SMOTE"))
BANDS = (
    (20, "Minimal Risk", "#1DB954"),
    (40, "Low Risk", "#F9C923"),
    (60, "Moderate Risk", "#F47C20"),
    (80, "High Risk", "#D9312B"),
    (float("inf"), "Critical Risk", "#7A0000"),
)
WEEKDAYS = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)
BLUE = "#086FC3"
PHILIPPINE_TIME = timezone(timedelta(hours=8))
_RENDER_LOCK = Lock()


def pdf_filename(exported_at: datetime) -> str:
    return exported_at.astimezone(PHILIPPINE_TIME).strftime(
        "Transaction-Results_%Y-%m-%d.pdf"
    )


def score_text(value: float) -> str:
    # JS Number.toFixed rounds the exact binary float, with ties away from zero.
    return str(
        Decimal.from_float(float(value)).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
    )


def risk_band(score: float) -> tuple:
    return next(band for band in BANDS if score < band[0])


def feature_text(feature: dict, derived: dict) -> str:
    name = feature["feature"]
    value = derived.get(name)
    # Derived inputs can arrive as floats; JavaScript displays integral numbers
    # without a decimal suffix in the on-screen SHAP narrative.
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if name == "day_of_week" and value is not None:
        return f"Manila weekday: {value} ({WEEKDAYS[int(value)]})"
    if name == "hour_of_day" and value is not None:
        return f"Manila hour: {value}"
    text = feature.get("readable_value") or name.replace("_", " ")
    for code in ("CASH_IN", "CASH_OUT", "TRANSFER", "PAYMENT", "DEBIT"):
        text = text.replace(code, code.lower().replace("_", " "))
    return text


def shap_summary(explanation: dict, model: str, derived: dict) -> str:
    item = explanation.get("models", {}).get(model)
    if not item:
        return "The SHAP explanation is unavailable. Retry it in the application, then download the results again."
    if not item.get("features"):
        return "No feature contributions are available for this record."

    def describe(direction):
        factors = sorted(
            (f for f in item["features"] if direction * f["contribution"] > 1e-9),
            key=lambda f: -direction * f["contribution"],
        )[:2]
        verb = "raised" if direction == 1 else "lowered"
        if not factors:
            return f"No transaction details meaningfully {verb} the score."
        sign = "+" if direction == 1 else "−"
        details = "; ".join(
            f"{feature_text(f, derived)} ({sign}{score_text(abs(f['contribution']) * 100)} percentage points)"
            for f in factors
        )
        return f"The main details that {verb} the score were: {details}."

    return (
        "SHAP shows how transaction details raised or lowered this model’s fraud score. "
        f"{describe(1)} {describe(-1)} "
        "These contributions explain the model’s score; they do not prove that a detail caused fraud."
    )


@lru_cache(maxsize=1)
def _register_fonts():
    folder = Path(matplotlib.get_data_path()) / "fonts/ttf"
    for name, file in (
        ("Receipt", "DejaVuSans.ttf"),
        ("ReceiptBold", "DejaVuSans-Bold.ttf"),
    ):
        pdfmetrics.registerFont(TTFont(name, str(folder / file)))
    pdfmetrics.registerFontFamily("Receipt", normal="Receipt", bold="ReceiptBold")


def _paragraph(text, width, size=11, color="#303438", bold=False):
    paragraph = Paragraph(
        text,
        ParagraphStyle(
            "receipt",
            fontName="ReceiptBold" if bold else "Receipt",
            fontSize=size,
            leading=size * 1.4,
            alignment=TA_CENTER,
            textColor=colors.HexColor(color),
            splitLongWords=True,
        ),
    )
    _, height = paragraph.wrap(width, 10000)
    return paragraph, height


def _draw_text(canvas, text, x, top, width, **kwargs):
    paragraph, height = _paragraph(text, width, **kwargs)
    paragraph.drawOn(canvas, x, top - height)
    return height


def _card(canvas, title, text, x, top, width, body_height, size=11):
    tab_width = min(width * 0.65, 285)
    canvas.setFillColor(colors.HexColor(BLUE))
    canvas.roundRect(
        x + (width - tab_width) / 2, top - 40, tab_width, 40, 12, fill=1, stroke=0
    )
    _draw_text(canvas, title, x, top - 8, width, size=11, color="#FFFFFF", bold=True)
    body_top = top - 30
    canvas.setFillColor(colors.HexColor("#EEEEEE"))
    canvas.roundRect(
        x, body_top - body_height - 2, width, body_height, 12, fill=1, stroke=0
    )
    canvas.setFillColor(colors.white)
    canvas.setStrokeColor(colors.HexColor("#DADDE1"))
    canvas.setLineWidth(0.6)
    canvas.roundRect(
        x, body_top - body_height, width, body_height, 12, fill=1, stroke=1
    )
    paragraph, height = _paragraph(text, width - 28, size)
    if height > body_height - 18:
        raise ValueError("Receipt PDF text exceeds its card")
    paragraph.drawOn(canvas, x + 14, body_top - (body_height + height) / 2)


def _gauge(canvas, score, threshold, x, top, width):
    height, bottom = 16, top - 16
    canvas.saveState()
    clip = canvas.beginPath()
    clip.roundRect(x, bottom, width, height, 8)
    canvas.clipPath(clip, stroke=0)
    for index, (_, _, color) in enumerate(BANDS):
        canvas.setFillColor(colors.HexColor(color))
        canvas.rect(
            x + index * width / 5, bottom, width / 5 + 0.1, height, fill=1, stroke=0
        )
    canvas.restoreState()
    threshold_x = x + width * min(100, max(0, threshold)) / 100
    canvas.setStrokeColor(colors.HexColor("#666666"))
    canvas.setLineWidth(2)
    canvas.line(threshold_x, bottom - 5, threshold_x, top + 5)
    canvas.setStrokeColor(colors.white)
    canvas.setLineWidth(0.8)
    canvas.line(threshold_x, bottom - 5, threshold_x, top + 5)
    canvas.setFillColor(colors.white)
    canvas.setStrokeColor(colors.HexColor("#A6ADB4"))
    canvas.setLineWidth(1.5)
    canvas.circle(
        x + width * min(100, max(0, score)) / 100, bottom + 8, 9, fill=1, stroke=1
    )
    for index, (_, label, _) in enumerate(BANDS):
        _draw_text(canvas, label, x + index * width / 5, bottom - 11, width / 5, size=9)
    _draw_text(
        canvas, f"Fraud threshold: {score_text(threshold)}%", x, top + 24, width, size=9
    )


def generate_pdf(result: dict) -> bytes:
    """Render exactly two landscape pages; fail rather than return clipped content."""
    # ReportLab's registered TrueType font objects cannot be used by concurrent
    # documents. This lock is independent of receipt/session lifecycle locks.
    with _RENDER_LOCK:
        return _render_pdf(result)


def _render_pdf(result: dict) -> bytes:
    _register_fonts()
    out = io.BytesIO()
    width, height = landscape(A4)
    canvas = Canvas(out, pagesize=(width, height), pageCompression=1)
    canvas.setTitle("Transaction Results - Benchmark RF and RF-SMOTE")
    canvas.setAuthor("Integritree")
    margin, gap = 36, 18
    content_width = width - 2 * margin
    reference = result["original"].get("reference")
    reference = reference.strip() if reference else ""
    for page, (model, name) in enumerate(MODELS, 1):
        score = result[model]["score"] * 100
        prediction = (
            "Fraudulent" if result[model]["predicted_label"] == 1 else "Legitimate"
        )
        prediction_color = "#C0392B" if prediction == "Fraudulent" else "#1DB954"
        _, band, band_color = risk_band(score)
        _draw_text(
            canvas,
            f"Transaction Record · {name}",
            margin,
            height - 25,
            content_width,
            size=18,
            bold=True,
        )
        reference_height = _draw_text(
            canvas,
            f'<font color="{BLUE}"><b>Transaction ID:</b></font> {escape(reference or "Not provided")}',
            margin,
            height - 54,
            content_width,
            size=10,
        )
        gauge_top = height - 54 - reference_height - 50
        _gauge(
            canvas,
            score,
            result["threshold"] * 100,
            margin + 9,
            gauge_top,
            content_width - 18,
        )
        top = gauge_top - 61
        half = (content_width - gap) / 2
        _card(
            canvas,
            "Prediction",
            f'<font color="{prediction_color}"><b>{prediction}</b></font>',
            margin,
            top,
            half,
            42,
        )
        _card(
            canvas,
            "Risk Score",
            f"<b>{score_text(score)} out of 100</b>",
            margin + half + gap,
            top,
            half,
            42,
        )
        top -= 92
        interpretation = (
            f'The {name} model classified the transaction as <font color="{prediction_color}"><b>{prediction}</b></font>, '
            f"with a fraud risk score of {score_text(score)}%, corresponding to a "
            f'<font color="{band_color}"><b>{band}</b></font> level.'
        )
        _, text_height = _paragraph(interpretation, content_width - 28)
        body_height = max(54, text_height + 22)
        _card(
            canvas,
            "Interpretation",
            interpretation,
            margin,
            top,
            content_width,
            body_height,
        )
        top -= 30 + body_height + 18
        summary = escape(
            shap_summary(result.get("explanation", {}), model, result["derived"])
        )
        available = top - 30 - 38
        for size in (11, 10.5, 10, 9.5):
            _, text_height = _paragraph(summary, content_width - 28, size)
            if text_height + 22 <= available:
                break
        else:
            raise ValueError("Receipt PDF explanation is too long to fit legibly")
        _card(
            canvas,
            "SHAP Explanation",
            summary,
            margin,
            top,
            content_width,
            max(82, text_height + 22),
            size,
        )
        _draw_text(
            canvas, f"{page} / 2", margin, 22, content_width, size=8, color="#687078"
        )
        canvas.showPage()
    canvas.save()
    return out.getvalue()
