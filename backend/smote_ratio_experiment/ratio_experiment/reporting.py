"""Offline reports that distinguish sampling effects from cutoff effects."""

from html import escape

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import pandas as pd

from investigation.diagnostics.reporting import (
    table as diagnostic_table,
    verify_offline_report,
)

CSS = """
*{box-sizing:border-box}body{margin:0;color:#182536;background:#f3f6f9;font:16px/1.6 system-ui,sans-serif}
main{max-width:1280px;margin:auto;padding:28px}header,section{background:white;padding:24px;margin:18px 0;border:1px solid #dce4ec;border-radius:12px}
h1{font-size:30px;line-height:1.2}h2{font-size:23px}a{color:#165a96}.notice{background:#fff5e7;padding:15px;border-left:4px solid #b35518}
.table-wrap{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:9px;border-bottom:1px solid #e0e6ec;text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}th{background:#edf2f7}
img{width:100%;height:auto}details{border:1px solid #dce4ec;border-radius:6px;padding:12px;margin:12px 0}summary{cursor:pointer;font-weight:600}.paired{display:grid;grid-template-columns:1fr 1fr;gap:20px}.paired article{min-width:0}code{overflow-wrap:anywhere}
@media(max-width:760px){main{padding:10px}header,section{padding:15px}.paired{grid-template-columns:1fr}h1{font-size:25px}}
"""
COLS = [
    "model",
    "threshold",
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


def table(frame, columns=None):
    return diagnostic_table(
        frame.map(lambda value: "undefined" if value is None else value), columns
    )


def page(title, content):
    return f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{escape(title)}</title><style>{CSS}</style></head><body><main>{content}</main></body></html>'


def save_pair_report(output, denominator, selected, fixed, tables, explanations):
    cutoff = float(selected.threshold.iloc[0])
    fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
    grid = tables["thresholds"]
    ax.plot(grid.threshold, grid.rf_f1, label="RF", color="#2364aa")
    ax.plot(
        grid.threshold,
        grid.rf_smote_f1,
        label=f"RF-SMOTE 1:{denominator}",
        color="#bd4f25",
    )
    ax.plot(
        grid.threshold,
        grid.mean_f1,
        label="Mean F1 (selection objective)",
        color="#368567",
        linestyle=":",
    )
    ax.axvline(
        cutoff, label=f"Shared cutoff {cutoff:.0%}", linestyle="--", color="#666"
    )
    ax.set(
        xlabel="Shared cutoff",
        ylabel="Validation F1",
        title=f"Ratio 1:{denominator}: same cutoff for both models",
    )
    ax.legend(fontsize=9)
    (output / "charts").mkdir()
    fig.savefig(output / "charts" / "thresholds.svg")
    plt.close(fig)
    cards = []
    for row in tables["sampled_cases"].itertuples():
        articles = []
        for model, label in (("rf", "RF"), ("rf_smote", f"RF-SMOTE 1:{denominator}")):
            item = explanations[(int(row.source_row_number), model)]
            prefix = f"explanations/{row.source_row_number}/{model}"
            articles.append(
                f'<article><h3>{label}: {item["output_value"]:.4f}</h3><p>{escape(item["narrative"])}</p><a href="{prefix}/waterfall.svg"><img loading="lazy" src="{prefix}/waterfall.svg" alt="{label} sampled case contributions"></a><a href="{prefix}/explanation.json">Explanation data</a></article>'
            )
        cards.append(
            f'<details><summary>Row {row.source_row_number} · {escape(row.type)} · {escape(row.outcome)}</summary><p>Actual label: {row.actual_label}; amount: {row.amount:,.2f} dataset units; simulation hour/day: {row.hour}/{row.day}.</p><div class="paired">{"".join(articles)}</div></details>'
        )
    groups = []
    for grouping, rows in tables["subgroups_selected"].groupby("grouping", sort=False):
        groups.append(
            f"<details><summary>{escape(grouping)}</summary>{table(rows, ['group', 'model', 'legitimate', 'fraud', 'fp', 'fn', 'precision', 'recall', 'f1', 'false_alarms_per_10000', 'precision_sparse', 'recall_sparse', 'false_alarms_per_10000_sparse'])}</details>"
        )
    downloads = "".join(
        f'<li><a href="tables/{name}.csv">{escape(name)} (CSV)</a></li>'
        for name in tables
    )
    content = f"""<header><h1>Ordinary SMOTE ratio 1:{denominator}</h1><p>Exploratory validation comparison · {len(selected):,} identical validation records · shared cutoff {cutoff:.0%}.</p></header>
<section><h2>Original 43% cutoff</h2><p>This keeps the cutoff fixed to isolate differences between fitted models.</p>{table(tables["overall_43"], COLS)}</section>
<section><h2>Selected shared cutoff: {cutoff:.0%}</h2><p>The mean F1 of RF and this candidate selects the cutoff. The existing 1:100 model is evaluated at the same cutoff as a reference and does not influence this selection.</p>{table(tables["overall_selected"], COLS)}<p>Average Precision is shown as pr_auc. Undefined values retain reasons in CSV. Metric supports below 20 are flagged; do not treat those subgroup rates as reliable findings.</p></section>
<section id="thresholds"><h2>Shared-cutoff selection</h2><img src="charts/thresholds.svg" alt="Validation F1 over shared thresholds"><p>Grid: 1%–100%; exact ties prefer closest to 50%, then the higher cutoff. This is validation development evidence, not a deployed setting.</p></section>
<section><h2>Paired outcomes</h2>{table(tables["outcomes_selected"])}<h2>Error patterns at the selected cutoff</h2><p>Amount bands are in dataset units. Hour/day are simulation indexes. Tables describe exploratory associations, not causes.</p>{"".join(groups)}</section>
<section id="shap"><h2>Sampled case explanations</h2><p>Up to five records per outcome/type group, seed 42, sorted row identities. Each case uses both models and the same 200 original-training background records. These samples are not global feature importance.</p>{"".join(cards)}</section>
<section><h2>Evidence files</h2><ul>{downloads}<li><a href="predictions_selected.parquet">Selected-cutoff validation predictions (Parquet)</a></li><li><a href="predictions_43.parquet">43% validation predictions (Parquet)</a></li><li><a href="model/training_audit.json">Sampling audit</a></li><li><a href="model/metadata.json">Model fingerprints</a></li></ul></section>"""
    (output / "report.html").write_text(
        page(f"SMOTE 1:{denominator}", content), encoding="utf-8"
    )
    verify_offline_report(output)


def save_summary(output, fixed, selected, candidates):
    # RF and the 1:100 reference repeat across fixed-cutoff pair tables; show once.
    fixed = fixed.drop_duplicates("model").reset_index(drop=True)
    order = ["RF", "RF-SMOTE 1:100 reference", *[f"RF-SMOTE 1:{d}" for d in candidates]]
    fixed = fixed.set_index("model").loc[order].reset_index()
    fixed.to_csv(output / "overall_43.csv", index=False, float_format="%.17g")
    selected.to_csv(output / "overall_selected.csv", index=False, float_format="%.17g")
    fixed.to_parquet(output / "overall_43.parquet", index=False)
    selected.to_parquet(output / "overall_selected.parquet", index=False)
    baseline = fixed.set_index("model").loc["RF-SMOTE 1:100 reference"]
    changes = []
    for denominator in candidates:
        candidate = fixed.set_index("model").loc[f"RF-SMOTE 1:{denominator}"]
        changes.append(
            {
                "ratio": f"1:{denominator}",
                "f1_at_43": candidate.f1,
                "f1_change_vs_100": candidate.f1 - baseline.f1,
                "fraud_detections_change_vs_100": int(candidate.tp - baseline.tp),
                "false_alarms_change_vs_100": int(candidate.fp - baseline.fp),
            }
        )
    changes = pd.DataFrame(changes)
    changes.to_csv(output / "changes_at_43.csv", index=False, float_format="%.17g")
    fig, ax = plt.subplots(figsize=(10, 4.5), layout="constrained")
    ax.barh(
        fixed.model,
        fixed.f1,
        color=["#2364aa" if name == "RF" else "#bd4f25" for name in fixed.model],
    )
    ax.set(
        xlabel="Validation F1 at the same 43% cutoff",
        title="Ordinary SMOTE intensity comparison",
    )
    ax.invert_yaxis()
    fig.savefig(output / "f1_at_43.svg")
    plt.close(fig)
    links = "".join(
        f'<li><a href="ratio_{d}/report.html">1:{d}: detailed comparison, error patterns and SHAP</a></li>'
        for d in candidates
    )
    content = f"""<header><h1>Ordinary SMOTE ratio comparison</h1><p>Completed · three new fits · validation only · no deployment.</p><p>RF and the existing 1:100 RF-SMOTE are unchanged references. Forest parameters, preprocessing, partitions, five neighbors and seed 42 are fixed.</p></header>
<section><h2>All models at the original 43% cutoff</h2>{table(fixed, COLS)}<img src="f1_at_43.svg" alt="F1 for all five models at 43 percent"><h3>Changes versus the existing 1:100 RF-SMOTE</h3>{table(changes)}<p>These fixed-cutoff comparisons separate the sampling change from cutoff selection. Counts are net differences, not necessarily the same individual records.</p></section>
<section><h2>Each candidate at its shared cutoff</h2><p>Each ratio has one cutoff, applied equally to RF, that candidate, and the existing 1:100 reference. Candidate comparisons may select different cutoffs; use the fixed-43% table to compare models without that change.</p>{table(selected, ["ratio_denominator", *COLS])}<p>Selection maximizes mean validation F1 for RF and the candidate on the 1%–100% grid. Exact ties prefer closest to 50%, then higher.</p></section>
<section><h2>Interpretation</h2><p class="notice">Exploratory results on an already inspected validation set. Three ratios and multiple cutoffs were examined; the best observed result is not an unbiased estimate of future performance. No model or cutoff is automatically promoted. Better F1 is not guaranteed.</p><p>AP denotes Average Precision. Metric denominators, undefined reasons and sparse flags are in the detailed CSVs. Subgroup patterns are descriptive and sampled SHAP explanations do not establish causation.</p></section>
<section><h2>Detailed evidence</h2><ul>{links}<li><a href="overall_43.csv">Fixed-cutoff metrics CSV</a></li><li><a href="overall_selected.csv">Shared-cutoff metrics CSV</a></li><li><a href="changes_at_43.csv">Changes CSV</a></li><li><a href="SUMMARY.md">Markdown summary</a></li><li><a href="manifest.json">Provenance manifest</a></li><li><a href="run.log">Run log</a></li></ul></section>"""
    summary = [
        "# Ordinary SMOTE ratio comparison",
        "",
        "Exploratory validation evidence; three new fits, no deployment or test-set evaluation.",
        "",
        "## All models at the original shared 43% cutoff",
        "",
        "| Model | F1 | Precision | Recall | TP | FP |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in fixed.itertuples():
        rendered = [
            "undefined" if pd.isna(v) else f"{v:.6f}"
            for v in (r.f1, r.precision, r.recall)
        ]
        summary.append(f"| {r.model} | {' | '.join(rendered)} | {r.tp} | {r.fp} |")
    summary += ["", "## Selected shared cutoffs", ""]
    for denominator in candidates:
        rows = selected[selected.ratio_denominator == denominator].set_index("model")
        row = rows.loc[f"RF-SMOTE 1:{denominator}"]
        summary.append(
            f"- 1:{denominator}: common cutoff {row.threshold:.0%}; RF F1 {rows.loc['RF'].f1:.6f}; candidate F1 {row.f1:.6f}. [Evidence](ratio_{denominator}/report.html)."
        )
    summary += [
        "",
        "Cutoffs maximize RF/candidate mean F1, with exact ties resolved by proximity to 50%, then higher. The 1:100 reference does not influence candidate selection.",
        "",
        "Compare sampler effects at 43%. Differences at selected cutoffs also include threshold selection. All results are exploratory on already inspected validation data. No candidate is automatically selected for deployment.",
        "",
        "See [complete report](report.html), [fixed metrics](overall_43.csv), [selected metrics](overall_selected.csv) and [manifest](manifest.json).",
    ]
    (output / "SUMMARY.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    (output / "report.html").write_text(
        page("Ordinary SMOTE ratio comparison", content), encoding="utf-8"
    )
    verify_offline_report(output)
