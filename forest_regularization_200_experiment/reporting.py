"""Tables and offline report for the completed 200-tree experiment."""

import html
from pathlib import Path

import pandas as pd


def table(frame):
    selected = frame[
        [
            "candidate",
            "model",
            "threshold",
            "precision",
            "recall",
            "f1",
            "pair_mean_f1",
            "pr_auc",
            "tp",
            "fp",
            "mean_time_abs_pp",
            "time_share_abs",
        ]
    ].copy()
    selected["candidate"] = selected.candidate.str.replace(
        "trees_200_depth_", "", regex=False
    ).str.replace("_leaf_", " / ", regex=False)
    selected["candidate"] = selected.candidate.replace({"20 / 1": "20 / 1 (baseline)"})
    selected["model"] = selected.model.map({"rf": "RF", "rf_smote": "RF-SMOTE"})
    for name in ("threshold", "precision", "recall", "time_share_abs"):
        selected[name] = selected[name].map(
            lambda value: "undefined" if pd.isna(value) else f"{value:.2%}"
        )
    for name in ("f1", "pair_mean_f1", "pr_auc", "mean_time_abs_pp"):
        selected[name] = selected[name].map(
            lambda value: "undefined" if pd.isna(value) else f"{value:.4f}"
        )
    selected.columns = [
        "Depth / leaf",
        "Model",
        "Cutoff",
        "Precision",
        "Recall",
        "F1",
        "Pair average F1",
        "PR-AUC (AP)",
        "Fraud detected",
        "False alarms",
        "Time contribution (pp)",
        "Time share",
    ]
    return selected


def report(output, results, manifest):
    frames = {}
    for name, key in (
        ("metrics", "rows"),
        ("time_shap", "shap"),
        ("time_sensitivity", "sweeps"),
        ("diagnostics", "diagnostics"),
    ):
        frames[name] = pd.DataFrame([row for result in results for row in result[key]])
        frames[name].to_csv(output / f"{name}.csv", index=False, float_format="%.17g")
    current = frames["metrics"].copy()
    current["pair_mean_f1"] = current.groupby(["candidate", "mode"]).f1.transform(
        "mean"
    )
    uniform = frames["time_shap"].query("cohort == 'uniform_validation'")
    combined = current.merge(
        uniform[["candidate", "model", "mean_time_abs_pp", "time_share_abs"]],
        on=["candidate", "model"],
        validate="many_to_one",
    )
    combined.to_csv(output / "comparison.csv", index=False, float_format="%.17g")
    pd.DataFrame([result["decision"] for result in results]).to_csv(
        output / "pair_decisions.csv", index=False, float_format="%.17g"
    )
    baseline = combined.query("candidate == 'trees_200_depth_20_leaf_1'")
    differences = combined.merge(
        baseline.drop(columns="candidate"),
        on=["model", "mode"],
        suffixes=("", "_baseline"),
        validate="many_to_one",
    )
    for name in (
        "threshold",
        "precision",
        "recall",
        "f1",
        "pair_mean_f1",
        "pr_auc",
        "tp",
        "fp",
        "mean_time_abs_pp",
        "time_share_abs",
    ):
        differences[f"delta_{name}"] = (
            differences[name] - differences[f"{name}_baseline"]
        )
    differences.to_csv(
        output / "changes_vs_200_tree_baseline.csv", index=False, float_format="%.17g"
    )
    previous_path = Path(manifest["settings"]["previous_results"])
    import json

    old_manifest = json.loads(
        (previous_path / "manifest.json").read_text(encoding="utf-8")
    )
    previous = pd.read_csv(
        previous_path / "comparison.csv", float_precision="round_trip"
    )
    previous["pair_mean_f1"] = previous.groupby(["candidate", "mode"]).f1.transform(
        "mean"
    )
    previous["n_estimators"] = 100
    previous["validation_csv_sha256"] = old_manifest["validation_csv_sha256"]
    previous.to_csv(
        output / "previous_100_tree_reference.csv", index=False, float_format="%.17g"
    )
    same_csv = (
        old_manifest["validation_csv_sha256"] == manifest["validation_csv_sha256"]
    )
    note = (
        f"Experimental validation on {manifest['rows']:,} CSV rows including {manifest['actual_fraud']:,} fraud labels. "
        "All six configurations use 200 trees. Three saved pairs are reused and six new models are trained on the original training split. "
        "RF-SMOTE uses one shared training-only ordinary-SMOTE sample at ratio 1:100. "
        "Shared cutoffs maximize average pair F1 on the 1%-100% grid; exact ties prefer closest to 50%, then higher. "
        "The cutoff is not necessarily optimal for each model individually. Cutoffs and metrics use the same validation data, not an independent test. "
        "No application model or cutoff has changed."
    )
    shap_note = (
        "Displayed SHAP measures use 200 uniform CSV records and the original 200-row training background. "
        "Time contribution (pp) = 100 x mean(|SHAP hour| + |SHAP day|). "
        "Time share = sum of absolute hour/day SHAP divided by sum of all absolute feature SHAP. "
        "Fraud and baseline-flagged cohort results are in time_shap.csv; cohorts overlap and baseline flags use 43%. "
        "All models share the same samples. Cutoff selection does not change SHAP values. "
        "The hour/day sweep is diagnostic substitution, not labeled accuracy evidence. "
        "CSV row numbers are positions, not original transaction identities. No provenance or overlap investigation was performed."
    )
    reference_note = (
        f"The separately listed 100-tree reference {'uses the same CSV fingerprint' if same_csv else 'uses a DIFFERENT CSV fingerprint'}. "
        "Tree count and depth settings differ, so differences cannot be attributed solely to depth. "
        "Only the uniform and actual-fraud sampling rules transfer directly; baseline-flagged membership changes with the baseline."
    )
    lines = ["# 200-tree regularization results", "", note, "", shap_note, ""]
    body = [
        "<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>",
        "<title>200-tree regularization results</title><style>body{font:16px system-ui;margin:2rem;color:#172838}table{border-collapse:collapse;font-size:13px}td,th{padding:7px;border:1px solid #ccc;white-space:nowrap}section{overflow:auto}p{max-width:105ch}</style>",
        "<h1>200-tree regularization results</h1><p>"
        + html.escape(note)
        + "</p><p>"
        + html.escape(shap_note)
        + "</p>",
    ]
    for mode, title in (
        ("selected", "Results at each pair's selected cutoff"),
        ("fixed_43", "Results at the fixed 43% cutoff"),
    ):
        display = table(combined[combined["mode"] == mode])
        lines += [
            f"## {title}",
            "",
            "| " + " | ".join(display.columns) + " |",
            "|" + "---|" * len(display.columns),
        ]
        lines += [
            "| " + " | ".join(map(str, row)) + " |"
            for row in display.itertuples(index=False, name=None)
        ]
        lines.append("")
        body += [
            "<h2>" + html.escape(title) + "</h2><section>",
            display.to_html(index=False, escape=True),
            "</section>",
        ]
    old = previous.query("mode == 'selected'")[
        [
            "candidate",
            "model",
            "threshold",
            "precision",
            "recall",
            "f1",
            "pair_mean_f1",
            "pr_auc",
            "tp",
            "fp",
            "mean_time_abs_pp",
            "time_share_abs",
        ]
    ]
    body += [
        "<h2>Earlier 100-tree reference</h2><p>"
        + html.escape(reference_note)
        + "</p><section>",
        old.to_html(
            index=False, float_format=lambda value: f"{value:.6f}", escape=True
        ),
        "</section><h2>Time sensitivity</h2><section>",
        frames["time_sensitivity"].to_html(
            index=False, float_format=lambda value: f"{value:.6f}"
        ),
        "</section><p>Confirm status=complete in manifest.json and status=passed in verification.json. Raw comparisons and all threshold choices are saved alongside this report.</p></html>",
    ]
    lines += [
        "## Earlier 100-tree reference",
        "",
        reference_note,
        "",
        "See previous_100_tree_reference.csv and report.html. Differences against the new 200-tree baseline are in changes_vs_200_tree_baseline.csv.",
        "",
        "Confirm completion in manifest.json and verification.json. PR-AUC means average precision.",
    ]
    (output / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "report.html").write_text("\n".join(body), encoding="utf-8")
