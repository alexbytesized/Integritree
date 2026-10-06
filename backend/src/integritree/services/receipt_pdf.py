"""Four-page receipt results and SHAP evaluations from a saved export snapshot."""

import io
import math
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

from integritree.ml.explainability import DECREASE_COLOR, FEATURE_LABELS, INCREASE_COLOR
from integritree.ml.features import FEATURE_COLUMNS

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
    """Render four landscape pages; fail rather than return clipped content."""
    # ReportLab's registered TrueType font objects cannot be used by concurrent
    # documents. This lock is independent of receipt/session lifecycle locks.
    with _RENDER_LOCK:
        return _render_pdf(result)


def contribution_ticks(values):
    """Nine tick-aligned gridlines containing zero and every signed pp value."""
    if any(not math.isfinite(value) or abs(value) > 100 for value in values):
        raise ValueError(
            "Receipt PDF contributions must be finite and within +/-100 pp"
        )
    low, high = min([0, *values]), max([0, *values])
    for step in (10, 20, 25):
        starts = [
            start
            for start in range(-100, 101 - 8 * step, step)
            if start % step == 0 and start <= low and start + 8 * step >= high
        ]
        if starts:
            start = min(starts, key=lambda value: (abs(value + 4 * step), value))
            return tuple(start + index * step for index in range(9))
    raise ValueError("Receipt PDF contribution range cannot fit")


def _header(canvas, name, reference, width, height, margin):
    content_width = width - 2 * margin
    _draw_text(
        canvas,
        f"Transaction Record \u00b7 {name}",
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
    return height - 54 - reference_height


def _footer(canvas, page, width, margin):
    _draw_text(
        canvas, f"{page} / 4", margin, 22, width - 2 * margin, size=8, color="#687078"
    )
    canvas.showPage()


def _pdf_features(item):
    features = item.get("features", [])
    names = [feature["feature"] for feature in features]
    # Incomplete explanations are unavailable, never padded with invented zeros.
    if len(names) != len(FEATURE_COLUMNS) or set(names) != set(FEATURE_COLUMNS):
        return None
    by_name = {feature["feature"]: feature for feature in features}
    return [by_name[name] for name in FEATURE_COLUMNS]


def _contribution_graph(canvas, features, x, top, width, bottom):
    values = [feature["contribution"] * 100 for feature in features]
    ticks = contribution_ticks(values)
    row_height = (top - bottom) / len(features)
    if row_height < 18:
        raise ValueError("Receipt PDF SHAP chart is too long to fit legibly")
    size = 9
    labels = [f"{value:+.3g} pp" for value in values]
    name_width = max(
        pdfmetrics.stringWidth(name, "Receipt", size)
        for name in FEATURE_LABELS.values()
    )
    end_padding = (
        max(pdfmetrics.stringWidth(label, "Receipt", size) for label in labels) + 10
    )
    name_right = x + name_width
    plot_left = name_right + 12 + end_padding
    plot_right = x + width - end_padding
    scale = (plot_right - plot_left) / (ticks[-1] - ticks[0])

    def position(value):
        return plot_left + (value - ticks[0]) * scale

    canvas.saveState()
    canvas.setFont("Receipt", size)
    for tick in ticks:
        tick_x = position(tick)
        canvas.setStrokeColor(colors.HexColor("#888888" if tick == 0 else "#DDDDDD"))
        canvas.setLineWidth(1 if tick == 0 else 0.5)
        canvas.line(tick_x, bottom, tick_x, top)
        canvas.setFillColor(colors.HexColor("#666666"))
        canvas.drawCentredString(tick_x, bottom - 15, str(tick))
    for row, (feature, value, label) in enumerate(zip(features, values, labels)):
        center = top - (row + 0.5) * row_height
        canvas.setFillColor(colors.HexColor("#303438"))
        canvas.drawRightString(
            name_right, center - size * 0.35, FEATURE_LABELS[feature["feature"]]
        )
        if value == 0:
            continue
        bar_width = abs(value) * scale
        bar_height = min(14, row_height * 0.6)
        color = colors.HexColor(INCREASE_COLOR if value > 0 else DECREASE_COLOR)
        canvas.setFillColor(color)
        canvas.roundRect(
            position(min(0, value)),
            center - bar_height / 2,
            bar_width,
            bar_height,
            min(3, bar_width / 2),
            fill=1,
            stroke=0,
        )
        if bar_width < 0.6:
            canvas.setStrokeColor(color)
            canvas.setLineWidth(0.6)
            canvas.line(
                position(value),
                center - bar_height / 2,
                position(value),
                center + bar_height / 2,
            )
        canvas.setFillColor(colors.HexColor("#303438"))
        if value < 0:
            canvas.drawRightString(position(value) - 5, center - size * 0.35, label)
        else:
            canvas.drawString(position(value) + 5, center - size * 0.35, label)
    canvas.setFillColor(colors.HexColor(DECREASE_COLOR))
    canvas.drawString(plot_left, bottom - 33, "\u2190 Decrease Fraud Risk Score")
    canvas.setFillColor(colors.HexColor(INCREASE_COLOR))
    canvas.drawRightString(plot_right, bottom - 33, "Increase Fraud Risk Score \u2192")
    canvas.restoreState()


def _shap_page(canvas, result, model, name, reference, width, height):
    margin, gap = 36, 18
    content_width = width - 2 * margin
    half = (content_width - gap) / 2
    header_bottom = _header(canvas, name, reference, width, height, margin)
    explanation = result.get("explanation", {})
    item = (
        explanation.get("models", {}).get(model, {})
        if explanation.get("status") == "computed"
        else {}
    )

    def percentage(key):
        value = item.get(key)
        return (
            f"{score_text(value * 100)}%"
            if isinstance(value, (int, float)) and math.isfinite(value)
            else "Unavailable"
        )

    top = item.get("top_positive_contributor") or {}
    contributor = (
        "No transaction details meaningfully increased the score."
        if top.get("status") == "no_positive_contributor"
        else FEATURE_LABELS.get(top.get("feature"), top.get("feature") or "Unavailable")
    )
    values = (percentage("base_value"), percentage("output_value"), escape(contributor))
    body_heights = [
        max(42, _paragraph(value, card_width - 28)[1] + 22)
        for value, card_width in zip(values, (half, half, content_width))
    ]
    contributor_top = 38 + 30 + body_heights[2]
    row_body = max(body_heights[:2])
    scores_top = contributor_top + gap + 30 + row_body
    chart_bottom = scores_top + 60
    chart_top = header_bottom - 16
    if chart_top - chart_bottom < 18 * len(FEATURE_COLUMNS):
        raise ValueError("Receipt PDF SHAP chart is too long to fit legibly")
    features = _pdf_features(item)
    if features is None:
        _draw_text(
            canvas,
            "The SHAP explanation is unavailable. Retry it in the application, then download the results again.",
            margin,
            (chart_top + chart_bottom) / 2,
            content_width,
        )
    else:
        _contribution_graph(
            canvas, features, margin, chart_top, content_width, chart_bottom
        )
    _card(canvas, "Reference Score", values[0], margin, scores_top, half, row_body)
    _card(
        canvas,
        "Output Score",
        values[1],
        margin + half + gap,
        scores_top,
        half,
        row_body,
    )
    _card(
        canvas,
        "Top Risk-Increasing Contributor",
        values[2],
        margin,
        contributor_top,
        content_width,
        body_heights[2],
    )


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
        header_bottom = _header(canvas, name, reference, width, height, margin)
        gauge_top = header_bottom - 50
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
        _footer(canvas, page * 2 - 1, width, margin)
        _shap_page(canvas, result, model, name, reference, width, height)
        _footer(canvas, page * 2, width, margin)
    canvas.save()
    return out.getvalue()
