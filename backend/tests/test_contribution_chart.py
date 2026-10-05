"""Signed SHAP presentation and cache migration, without loading a model."""

from copy import deepcopy

import matplotlib
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from integritree.ml import explainability as charts

matplotlib.use("Agg")


def explanation(values):
    return {
        "model": "rf",
        "transaction_id": "synthetic-transaction",
        "base_value": 0.5,
        "output_value": 0.5 + sum(values),
        "features": [
            {
                "feature": charts.FEATURE_COLUMNS[i],
                "readable_value": f"Feature {i} = {i}",
                "contribution": value,
            }
            for i, value in enumerate(values)
        ],
    }


@pytest.mark.parametrize(
    "values",
    [
        [0.12, -0.08, 0.08, -1e-10, 1e-10, 0, 0.02, 0, 0, 0, 0],
        [0.01] * 11,
        [-0.01] * 11,
        [0.0] * 11,
        [0, 0.4, -0.4, 1, -1, 1e-10, -1e-10, 0, 0, 0, 0],
    ],
)
def test_chart_preserves_signed_effects_and_shows_all_features(
    tmp_path, monkeypatch, values
):
    item = explanation(values)
    item["features"].reverse()  # Presentation order must not depend on payload order.
    original = deepcopy(item)
    figures = []
    savefig = Figure.savefig

    def capture(figure, *args, **kwargs):
        figures.append(figure)
        return savefig(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture)
    output = tmp_path / "chart.svg"
    charts.waterfall(item, output, display_label="1")
    axis = figures[0].axes[0]
    order = list(range(11))
    nonzero = [i for i in order if values[i] != 0]
    assert len(axis.patches) == len(nonzero)
    assert [bar.get_x() for bar in axis.patches] == pytest.approx(
        [min(0, values[i] * 100) for i in nonzero]
    )
    assert [bar.get_width() for bar in axis.patches] == pytest.approx(
        [abs(values[i]) * 100 for i in nonzero]
    )
    assert [tick.get_text() for tick in axis.get_yticklabels()] == [
        charts.FEATURE_LABELS[charts.FEATURE_COLUMNS[i]] for i in order
    ]
    for bar, i in zip(axis.patches, nonzero):
        color = matplotlib.colors.to_hex(bar.get_facecolor())
        assert color == ("#a00000" if values[i] > 0 else "#009900")
    labels = [text.get_text() for text in axis.texts[: len(nonzero)]]
    assert labels == [f"{values[i] * 100:+.3g} pp" for i in nonzero]
    assert axis.get_xlim() == (-100, 100)
    assert list(axis.get_xticks()) == list(range(-100, 101, 10))
    assert axis.bbox.width == pytest.approx(1600)
    assert axis.bbox.height == pytest.approx(440)
    assert all(tick.get_fontweight() == "normal" for tick in axis.get_yticklabels())
    assert all(tick.get_fontweight() == "normal" for tick in axis.get_xticklabels())
    FigureCanvasAgg(figures[0]).draw()
    for text, row in zip(axis.texts[: len(nonzero)], nonzero):
        bounds = text.get_window_extent(figures[0].canvas.get_renderer())
        assert bounds.x0 >= 0 and bounds.x1 <= figures[0].bbox.width
        name_bounds = axis.get_yticklabels()[row].get_window_extent(
            figures[0].canvas.get_renderer()
        )
        assert bounds.x0 > name_bounds.x1

    svg = output.read_text(encoding="utf-8")
    assert charts.CHART_TITLE in svg
    assert "Benchmark RF | 1" in svg
    assert "Reference" in svg and "50.00%" in svg
    assert "Decrease Fraud Risk Score" in svg and "Increase Fraud Risk Score" in svg
    assert item == original
    assert item["base_value"] + sum(values) == pytest.approx(item["output_value"])


def test_versioned_cache_ignores_waterfalls_and_tracks_values_and_labels(
    tmp_path, monkeypatch
):
    legacy = tmp_path / "waterfall.svg"
    legacy.write_text("old waterfall", encoding="utf-8")
    item = explanation([0.1, -0.05])
    first = charts.cached_contribution_chart(item, tmp_path)
    rendered_at = first.stat().st_mtime_ns
    assert first.name.startswith(charts.CHART_VERSION)
    assert charts.cached_contribution_chart(item, tmp_path) == first
    assert first.stat().st_mtime_ns == rendered_at
    assert legacy.read_text() == "old waterfall"
    display = charts.cached_contribution_chart(item, tmp_path, display_label="1")
    assert display != first
    item["features"][0]["readable_value"] = "Updated input = 1"
    changed = charts.cached_contribution_chart(item, tmp_path)
    assert changed != first
    assert item["features"][0]["readable_value"] == "Updated input = 1"
    modal = charts.cached_contribution_chart(item, tmp_path, layout="modal")
    assert modal != changed
    assert "Benchmark RF |" not in modal.read_text(encoding="utf-8")
    assert "Reference" not in modal.read_text(encoding="utf-8")
    assert charts.CHART_TITLE not in modal.read_text(encoding="utf-8")
    assert "Hour of the Day" not in modal.read_text(encoding="utf-8")
    assert "Decrease Fraud Risk Score" not in modal.read_text(encoding="utf-8")
    monkeypatch.setattr(charts, "CHART_VERSION", "contribution_bars_test_v2")
    assert charts.cached_contribution_chart(item, tmp_path) != changed


def test_standalone_summary_and_zero_rows(tmp_path):
    item = explanation([0] * 11)
    item["top_positive_contributor"] = {
        "status": "no_positive_contributor",
        "feature": None,
    }
    chart = charts.cached_contribution_chart(item, tmp_path)
    svg = chart.read_text(encoding="utf-8")
    assert "No transaction details meaningfully increased the score." in svg
    assert "0 pp" not in svg and 'id="shap-' not in svg
    item["features"][1]["contribution"] = 0.07
    item["output_value"] = 0.57
    item["top_positive_contributor"] = {"status": "available", "feature": "day_of_week"}
    changed = charts.cached_contribution_chart(item, tmp_path)
    assert changed != chart
    assert "57.00%" in changed.read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="layout"):
        charts.cached_contribution_chart(item, tmp_path, layout="invalid")


def test_modal_pixel_geometry_and_alignment(tmp_path, monkeypatch):
    figures = []
    savefig = Figure.savefig

    def capture(figure, *args, **kwargs):
        figures.append(figure)
        return savefig(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture)
    charts.waterfall(
        explanation([0.4, -0.4, 1, -1] + [0] * 7),
        tmp_path / "modal.svg",
        layout="modal",
    )
    figure = figures[0]
    axis = figure.axes[0]
    assert figure.bbox.width == pytest.approx(1800)
    assert figure.bbox.height == pytest.approx(488)
    assert axis.bbox.width == pytest.approx(1600)
    assert len(axis.get_yticklabels()) == 0
    assert axis.get_title() == ""
    for row in range(11):
        x, y = axis.transData.transform((0, row))
        assert x == pytest.approx(900)
        assert figure.bbox.height - y == pytest.approx(8 + 20 + 40 * row)
    assert [line.get_xdata()[0] for line in axis.lines] == list(range(-100, 101, 10))
