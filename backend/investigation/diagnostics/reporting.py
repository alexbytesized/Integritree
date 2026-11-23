"""Portable, offline report rendering; all narrative is tied to measured evidence."""

from html import escape

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .analysis import MODELS

COLORS = {"rf": "#2364aa", "rf_smote": "#bd4f25"}


def save_charts(output, frame, grid, subgroups):
    directory = output / "charts"
    directory.mkdir()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    for model in MODELS:
        ax.plot(grid.threshold, grid[f"{model}_f1"], label=model, color=COLORS[model])
    ax.axvline(
        float(frame.threshold.iloc[0]),
        linestyle="--",
        color="#666",
        label="Saved cutoff",
    )
    ax.set(
        xlabel="Diagnostic threshold",
        ylabel="Validation F1",
        title="Threshold sensitivity (1%–100%)",
    )
    ax.legend()
    fig.savefig(directory / "thresholds.svg")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    for actual in (0, 1):
        for column, model in enumerate(MODELS):
            ax = axes[actual, column]
            part = frame[frame.actual_label == actual]
            for outcome, group in part.groupby("outcome", sort=True):
                # Each outcome has its own normalization, so rare errors remain visible.
                ax.hist(
                    group[f"{model}_risk_score"],
                    bins=np.linspace(0, 1, 21),
                    weights=np.full(len(group), 1 / len(group)),
                    histtype="step",
                    linewidth=1.5,
                    label=f"{outcome.replace('_', ' ')} (n={len(group):,})",
                )
            ax.set(
                title=f"{model}: actual {'fraud' if actual else 'legitimate'}",
                xlabel="Saved score",
                ylabel="Fraction within outcome",
            )
            ax.legend(fontsize=6.5)
    fig.savefig(directory / "scores.svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    data = subgroups[subgroups.grouping == "type"].pivot(
        index="group", columns="model", values="false_alarms_per_10000"
    )
    positions = np.arange(len(data))
    for i, model in enumerate(MODELS):
        ax.bar(
            positions + (i - 0.5) * 0.36,
            data[model],
            0.36,
            color=COLORS[model],
            label=model,
        )
    ax.set_xticks(positions, data.index)
    ax.set(
        ylabel="False alarms per 10,000 legitimate transactions",
        title="False-alarm rates by transaction type (descriptive)",
    )
    ax.legend()
    fig.savefig(directory / "type_rates.svg")
    plt.close(fig)


def table(frame, columns=None):
    if columns is not None:
        frame = frame.loc[:, columns]
    return (
        '<div class="table-wrap">'
        + frame.to_html(
            index=False,
            border=0,
            escape=True,
            na_rep="—",
            float_format=lambda v: f"{v:.6g}",
        )
        + "</div>"
    )


def write_report(output, context, tables, explanations):
    notes = context["findings"]
    links = [(f"tables/{name}.csv", name.replace("_", " ")) for name in tables]
    downloads = "".join(
        f'<li><a href="{escape(path)}">{escape(label)} (CSV)</a></li>'
        for path, label in links
    )
    cards = []
    for record in context["samples"]:
        row_id = int(record["source_row_number"])
        details = []
        for model in MODELS:
            item = explanations[(row_id, model)]
            detail_path = f"explanations/{row_id}/{model}"
            details.append(
                f"<article><h4>{escape(model)} · score {item['output_value']:.4f}</h4>"
                f"<p>{escape(item['narrative'])}</p>"
                f'<img loading="lazy" src="{detail_path}/waterfall.svg" '
                f'alt="{escape(model)} signed feature contributions for source row {row_id}">'
                f'<a href="{detail_path}/explanation.json">Explanation data</a></article>'
            )
        cards.append(
            f"<details><summary>Row {row_id} · {escape(record['type'])} · "
            f"{escape(record['outcome'].replace('_', ' '))}</summary>"
            f"<p>Actual label: {int(record['actual_label'])}; amount: {record['amount']:,.2f}; "
            f"simulation hour/day: {int(record['hour'])}/{int(record['day'])}.</p>"
            '<div class="paired">' + "".join(details) + "</div></details>"
        )
    overview_columns = [
        "model",
        "rows",
        "tp",
        "fp",
        "fn",
        "tn",
        "precision",
        "recall",
        "f1",
        "mcc",
        "pr_auc",
        "false_alarms_per_10000",
    ]
    sub_columns = [
        "group",
        "model",
        "legitimate",
        "fraud",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1",
        "mcc",
        "pr_auc",
        "false_alarms_per_10000",
        "precision_sparse",
        "recall_sparse",
        "false_alarms_per_10000_sparse",
    ]
    sections = []
    for grouping, group in tables["subgroups"].groupby("grouping", sort=False):
        sections.append(
            f"<details><summary>{escape(grouping)}</summary>"
            + table(group, sub_columns)
            + "</details>"
        )
    findings_html = "".join(f"<li>{escape(note)}</li>" for note in notes)
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RF / RF-SMOTE validation investigation</title>
<style>
*{{box-sizing:border-box}}body{{margin:0;background:#f4f6f8;color:#182536;font:16px/1.6 system-ui,sans-serif}}
main{{max-width:1250px;margin:auto;padding:30px}}header,section{{background:white;border:1px solid #dde4eb;border-radius:12px;padding:24px;margin:20px 0}}
h1{{font-size:30px;line-height:1.2}}h2{{font-size:22px}}h3{{font-size:18px}}a{{color:#165a96}}.badge{{color:#235d42;background:#e9f5ed;padding:5px 10px;border-radius:5px}}
.notice{{border-left:4px solid #bd4f25;padding:10px 18px;background:#fff7ed}}.table-wrap{{overflow:auto;max-height:650px}}
table{{border-collapse:collapse;font-size:13px;width:100%}}th,td{{text-align:right;padding:9px;border-bottom:1px solid #e4e9ee;white-space:nowrap}}th{{background:#edf2f7;position:sticky;top:0}}th:first-child,td:first-child{{text-align:left}}
img{{width:100%;height:auto}}details{{margin:12px 0;border:1px solid #d8e0e7;border-radius:8px;padding:12px}}summary{{cursor:pointer;font-weight:600}}.paired{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}article{{min-width:0}}nav a{{margin-right:15px}}.muted{{color:#526274}}code{{overflow-wrap:anywhere}}
@media(max-width:760px){{main{{padding:12px}}section,header{{padding:16px}}.paired{{grid-template-columns:1fr}}h1{{font-size:25px}}}}
</style></head><body><main>
<header><span class="badge">Completed diagnostic report · validation only</span>
<h1>Where do RF and RF-SMOTE differ?</h1>
<p>Run <code>{escape(context["run_id"])}</code> · {escape(context["created_at"])}</p>
<p>{context["rows"]:,} validation transactions · saved cutoff {context["threshold"]:.0%} · F1 is the priority for later recalibration discussions.</p>
<nav><a href="#findings">Findings</a><a href="#patterns">Patterns</a><a href="#thresholds">Thresholds</a><a href="#shap">Case explanations</a><a href="#downloads">Data & provenance</a></nav>
</header>
<section id="findings"><h2>Measured findings</h2><ul>{findings_html}</ul>
<p class="notice">No model was retrained, no cutoff was deployed, and no test-set evaluation was performed. Differences describe this inspected validation set; they do not establish generalization or causation.</p>
<h3>Overall performance</h3>{table(tables["overall"], overview_columns)}
<p>PR-AUC is Average Precision. An em dash denotes an undefined value; CSV exports retain the reason.</p>
<h3>Paired outcomes</h3>{table(tables["outcomes"])}
<p><a href="tables/outcomes_by_type.csv">Outcomes by transaction type</a> · <a href="tables/outcomes_by_amount.csv">Outcomes by amount</a></p></section>
<section id="patterns"><h2>Error patterns and support</h2>
<p>Counts and rates answer different questions. Sparse flags mark metric support below {context["minimum_denominator"]}; flagged metrics are excluded from highlighted findings. CSV tables include every metric's support and undefined-value reason.</p>
<p>Precision uses predicted positives; recall and AP use actual positives; F1 uses 2TP+FP+FN; MCC support is its smallest marginal count; false-alarm rate uses actual negatives. These supports are not confidence intervals. Amount uses dataset units, and hour/day are simulation positions, not real dates.</p>
<img src="charts/type_rates.svg" alt="False-alarm rates by transaction type for both models">
{"".join(sections)}
<h3>Scores within each outcome</h3><p>Every curve is normalized within its own outcome group. This makes rare mistakes visible; curve height is not population prevalence.</p>
<img src="charts/scores.svg" alt="Saved score distributions grouped by actual label and paired outcome">
</section>
<section id="thresholds"><h2>Threshold effects</h2><p>Alternative thresholds are exploratory comparisons, not selected replacements. Whole-percentage F1 ties prefer proximity to 50%, then the higher cutoff.</p>
<img src="charts/thresholds.svg" alt="Validation F1 across the 100 diagnostic cutoffs">
<h3>Each model's best F1 on the existing grid</h3>{table(tables["best_grid_f1"])}
<h3>Comparable false-alarm budgets</h3>{table(tables["false_alarm_budgets"])}
<p>Budgets equal each model's current false-alarm count. Comparisons maximize detections without exceeding the budget, then minimize false alarms and prefer the higher cutoff. Equal scores remain together; counts are attained, not interpolated. A no-alert point has no numerical cutoff.</p>
</section>
<section id="shap"><h2>Paired case explanations</h2><p>{len(context["samples"])} sampled transactions, each explained with both fitted models. Up to {context["sample_per_group"]} records per outcome/type cell, seed {context["seed"]}, with a shared {context["background_size"]}-record original-training background.</p>
<p class="notice">These are deliberately stratified case studies, not population-wide feature importance. SHAP explains model score contributions; it does not establish why SMOTE caused a mistake. Correct cases are sampled controls, not individually matched neighbors.</p>
<details><summary>Sampling coverage</summary>{table(tables["sample_coverage"])}</details>{"".join(cards)}
</section>
<section id="downloads"><h2>Data and reproducibility</h2><ul>{downloads}
<li><a href="validation_details.parquet">All joined validation details (Parquet)</a></li>
<li><a href="SUMMARY.md">Markdown summary</a></li><li><a href="manifest.json">Manifest, input hashes, settings and verification</a></li>
<li><a href="run.log">Execution log</a></li></ul><p class="muted">Input artifacts were verified unchanged. This report uses local assets only and can be opened offline. Share the entire run folder to preserve all links.</p></section>
</main></body></html>"""
    (output / "report.html").write_text(html, encoding="utf-8")
    summary = [
        "# RF / RF-SMOTE validation investigation",
        "",
        f"Run: {context['run_id']}",
        "",
        f"Validation rows: {context['rows']:,}; saved cutoff: {context['threshold']:.2f}.",
        "",
        *[f"- {note}" for note in notes],
        "",
        "See [the offline report](report.html) for subgroup support, threshold comparisons and paired SHAP cases.",
        "",
        "No retraining, deployment changes, or test-set evaluation. SHAP case samples are not global importance estimates.",
    ]
    (output / "SUMMARY.md").write_text("\n".join(summary) + "\n", encoding="utf-8")


def verify_offline_report(output):
    """Check generated local references and reject accidentally introduced remote assets."""
    from html.parser import HTMLParser

    class References(HTMLParser):
        def handle_starttag(self, tag, attrs):
            for key, value in attrs:
                if key not in ("href", "src") or not value or value.startswith("#"):
                    continue
                if ":" in value or value.startswith("//"):
                    raise ValueError(f"Nonlocal report reference: {value}")
                target = (output / value).resolve()
                if not target.is_relative_to(output.resolve()) or not target.is_file():
                    raise ValueError(f"Missing or escaped report reference: {value}")

    References().feed((output / "report.html").read_text(encoding="utf-8"))
