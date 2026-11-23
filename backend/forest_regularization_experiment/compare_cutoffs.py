"""Select one shared validation-F1 cutoff per saved pair; never fit or deploy models."""

import argparse
import html
import inspect
import json
import sys
from datetime import UTC, datetime
from fractions import Fraction
from importlib.metadata import version
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "src"))

import numpy as np
import pandas as pd

from integritree.config import EvaluationConfig
from integritree.ml.data import file_sha256, write_json
from integritree.ml.evaluation import binary_labels, metrics, score_vector
from integritree.ml.research import new_run
from integritree.ml.threshold_search import percent_grid_thresholds

MODELS = ("rf", "rf_smote")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify_hashes(inventory):
    for path, expected in inventory.items():
        if file_sha256(Path(path)) != expected:
            raise ValueError(f"Fingerprint mismatch: {path}")


def source_inventory(source):
    source = source.resolve()
    manifest = read_json(source / "manifest.json")
    if (
        manifest.get("kind") != "isolated_forest_regularization"
        or manifest.get("status") != "complete"
        or manifest.get("test_used") is not False
        or manifest.get("source_preservation_verified") is not True
    ):
        raise ValueError("A completed validation-only forest experiment is required")
    inventory = {str(source / "manifest.json"): file_sha256(source / "manifest.json")}
    for name, expected in manifest["files"].items():
        target = (source / name).resolve()
        if not target.is_relative_to(source):
            raise ValueError("Source file escapes its experiment directory")
        inventory[str(target)] = expected
    verify_hashes(inventory)
    return manifest, inventory


def verify_grid(labels, scores, decision, grid):
    """Independent brute-force count sweep and exact rational tie handling."""
    y = binary_labels(labels)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Both validation classes are required")
    if list(grid.threshold_percent) != list(range(1, 101)):
        raise ValueError("Expected the complete 1..100 percent grid")
    scored = {k: score_vector(scores[k], len(y)) for k in MODELS}
    candidates = []
    for row in grid.itertuples(index=False):
        percent = int(row.threshold_percent)
        cutoff = percent / 100
        if row.threshold != cutoff:
            raise ValueError("Threshold grid is inconsistent")
        f1 = []
        for kind in MODELS:
            predicted = scored[kind] >= cutoff
            tp = int(np.count_nonzero(predicted & (y == 1)))
            fp = int(np.count_nonzero(predicted & (y == 0)))
            fn = int(np.count_nonzero(~predicted & (y == 1)))
            tn = int(np.count_nonzero(~predicted & (y == 0)))
            for name, value in {"tp": tp, "fp": fp, "fn": fn, "tn": tn}.items():
                if getattr(row, f"{kind}_{name}") != value:
                    raise ValueError(
                        f"Threshold count mismatch: {kind}/{percent}/{name}"
                    )
            fraction = Fraction(2 * tp, 2 * tp + fp + fn)
            np.testing.assert_allclose(
                getattr(row, f"{kind}_f1"), float(fraction), rtol=0, atol=1e-15
            )
            f1.append(fraction)
        mean = sum(f1) / 2
        np.testing.assert_allclose(row.mean_f1, float(mean), rtol=0, atol=1e-15)
        candidates.append((mean, -abs(percent - 50), percent))
    expected = max(candidates)
    if (
        decision["threshold"] != expected[2] / 100
        or Fraction(decision["mean_f1_fraction"]) != expected[0]
        or decision["candidate_count"] != 100
    ):
        raise ValueError("Shared cutoff violates exact mean-F1 selection or tie policy")
    return {
        "status": "passed",
        "thresholds_verified": 100,
        "selected_percent": expected[2],
        "mean_f1_fraction": str(expected[0]),
        "method": "independent direct classification at every cutoff",
    }


def metric_row(y, scores, threshold, policy, candidate, kind, mode):
    result = metrics(y, scores, threshold, policy)
    row = {
        "candidate": candidate,
        "model": kind,
        "mode": mode,
        "threshold": threshold,
        "rows": len(y),
        "actual_fraud": int(np.sum(y)),
    }
    for name, item in result["metrics"].items():
        row[name], row[f"{name}_undefined_reason"] = item["value"], item["reason"]
    for name in ("tp", "fp", "tn", "fn"):
        row[name] = result["confusion_matrix"][name]
    row["false_alarms_per_10000_legitimate"] = (
        10000 * row["fp"] / (row["fp"] + row["tn"])
    )
    return row


def save_table(frame, output, name):
    frame.to_csv(output / f"{name}.csv", index=False, float_format="%.17g")
    frame.to_parquet(output / f"{name}.parquet", index=False)


def render(output, source, summary, selected):
    note = (
        "One shared cutoff per RF/RF-SMOTE pair, chosen by maximum mean validation F1 "
        "on the 1%..100% grid. Exact ties prefer closest to 50%, then higher. "
        "This is exploratory selection on previously inspected validation data, not an independent test. "
        "Scores, probability-space SHAP on the same records/background, and Average Precision are unchanged. "
        "No models were retrained and no application cutoff was changed."
    )
    lines = [
        "# Shared-cutoff comparison",
        "",
        note,
        "",
        "| Depth / leaf | Selected cutoff | RF F1 | RF-SMOTE F1 | Pair mean F1 at 43% | Pair mean F1 selected |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.max_depth} / {row.min_samples_leaf} | {row.threshold:.0%} | "
            f"{row.rf_f1_selected:.6f} | {row.rf_smote_f1_selected:.6f} | "
            f"{row.mean_f1_fixed:.6f} | {row.mean_f1_selected:.6f} |"
        )
    lines += [
        "",
        (
            "'Best' means best shared mean F1 among these 100 grid cutoffs. "
            "It is not a separate optimum for each model, an optimum over every possible real-valued "
            "threshold, or proof of future performance."
        ),
        "",
        (
            f"Source run: `{source.name}`. Full metric tables, all threshold curves, "
            "selected decisions and verification are in this folder."
        ),
    ]
    (output / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    display = summary[
        [
            "max_depth",
            "min_samples_leaf",
            "threshold",
            "rf_f1_selected",
            "rf_smote_f1_selected",
            "mean_f1_fixed",
            "mean_f1_selected",
        ]
    ].copy()
    display["threshold"] = display.threshold.map(lambda v: f"{v:.0%}")
    display.columns = [
        "Depth",
        "Minimum leaf",
        "Shared cutoff",
        "RF F1",
        "RF-SMOTE F1",
        "Mean F1 at 43%",
        "Mean F1 selected",
    ]
    detail = selected[
        [
            "candidate",
            "model",
            "threshold",
            "precision",
            "recall",
            "f1",
            "mcc",
            "pr_auc",
            "tp",
            "fp",
        ]
    ].copy()
    for name in ("threshold", "precision", "recall"):
        detail[name] = detail[name].map(
            lambda v: "undefined" if pd.isna(v) else f"{v:.2%}"
        )
    detail.columns = [
        "Configuration",
        "Model",
        "Shared cutoff",
        "Precision",
        "Recall",
        "F1",
        "MCC",
        "PR-AUC (AP)",
        "Fraud detected",
        "False alarms",
    ]
    body = (
        "<!doctype html><html lang='en'><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>Shared-cutoff comparison</title><style>body{font:16px system-ui;margin:2rem;color:#172838}"
        "table{border-collapse:collapse;font-size:14px}td,th{padding:8px;border:1px solid #ccc}"
        "section{overflow:auto;margin-bottom:2rem}p{max-width:85ch}</style><h1>Shared-cutoff comparison</h1><p>"
        + html.escape(note)
        + "</p><h2>Pair comparison</h2><section>"
        + display.to_html(index=False, float_format=lambda v: f"{v:.6f}", escape=True)
        + "</section><h2>Metrics at each pair's selected cutoff</h2><section>"
        + detail.to_html(index=False, float_format=lambda v: f"{v:.6f}", escape=True)
        + "</section><p>Read manifest.json for completion status. Separate fixed-43% and selected "
        "tables and all 600 grid rows are retained. Newly flagged sets can differ; no new flagged-set "
        "SHAP summary was computed.</p></html>"
    )
    (output / "report.html").write_text(body, encoding="utf-8")


def run(source, output_root, run_id=None):
    source = Path(source).resolve()
    source_manifest, inventory = source_inventory(source)
    plan = read_json(source / "plan.json")
    if plan["candidates"] != source_manifest["plan"]:
        raise ValueError("Source plan and manifest disagree")
    fixed_cutoff = plan["fixed_configuration"]["scoring"]["threshold"]
    if fixed_cutoff != 0.43:
        raise ValueError("This follow-up compares against the fixed 43% experiment")
    policy = EvaluationConfig.model_validate(plan["fixed_configuration"]["evaluation"])
    output = new_run(Path(output_root), "paired_cutoffs", run_id)
    if output.is_relative_to(source):
        raise ValueError("Output must be separate from the completed source run")
    implementation = {
        str(p): file_sha256(p)
        for p in (
            Path(__file__),
            Path(inspect.getfile(percent_grid_thresholds)),
            Path(inspect.getfile(metrics)),
        )
    }
    manifest = {
        "schema_version": 1,
        "kind": "isolated_paired_cutoff_analysis",
        "status": "running",
        "created_at": datetime.now(UTC).isoformat(),
        "source": str(source),
        "new_fits": 0,
        "test_used": False,
        "source_sha256": inventory,
        "implementation_sha256": implementation,
        "selection": {
            "objective": "mean_validation_f1",
            "grid_percent": list(range(1, 101)),
            "tie_policy": "closest_to_50_then_higher",
            "score_ties": "fraud",
        },
        "packages": {
            p: version(p) for p in ("numpy", "pandas", "scikit-learn", "pyarrow")
        },
        "completed": [],
    }
    write_json(output / "manifest.json", manifest)
    try:
        fixed_rows, chosen_rows, summary, checks, all_grids, boundary = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        identity = labels = None
        fixed_source = pd.read_csv(source / "metrics.csv", float_precision="round_trip")
        for item in plan["candidates"]:
            candidate = item["candidate"]
            if Path(candidate).name != candidate or candidate in (".", ".."):
                raise ValueError("Invalid candidate identifier")
            frame = pd.read_parquet(source / candidate / "predictions.parquet")
            if (
                frame.source_row_number.isna().any()
                or not frame.source_row_number.is_unique
            ):
                raise ValueError("Missing or duplicate validation identities")
            y = binary_labels(frame.actual_label)
            if identity is None:
                identity, labels = frame.source_row_number.to_numpy(), y.copy()
            else:
                np.testing.assert_array_equal(frame.source_row_number, identity)
                np.testing.assert_array_equal(y, labels)
            scores = {k: score_vector(frame[f"{k}_score"], len(y)) for k in MODELS}
            for kind in MODELS:
                np.testing.assert_array_equal(
                    frame[f"{kind}_predicted_label"], scores[kind] >= fixed_cutoff
                )
            decision, grid = percent_grid_thresholds(
                y, scores["rf"], scores["rf_smote"]
            )
            check = verify_grid(y, scores, decision, grid)
            checks.append({"candidate": candidate} | check)
            threshold = decision["threshold"]
            print(
                f"{candidate}: shared cutoff {threshold:.0%}; all 100 choices independently verified",
                flush=True,
            )
            folder = output / candidate
            folder.mkdir()
            write_json(
                folder / "decision.json",
                decision | {"candidate": candidate, "verification": check},
            )
            save_table(grid, folder, "thresholds")
            all_grids.append(grid.assign(candidate=candidate))
            selected_predictions = frame.copy()
            selected_predictions["shared_threshold"] = threshold
            fixed_pair, selected_pair = {}, {}
            for kind in MODELS:
                fixed = metric_row(
                    y, scores[kind], fixed_cutoff, policy, candidate, kind, "fixed_43"
                )
                chosen = metric_row(
                    y, scores[kind], threshold, policy, candidate, kind, "selected"
                )
                prior = fixed_source[
                    (fixed_source.candidate == candidate) & (fixed_source.model == kind)
                ]
                if len(prior) != 1:
                    raise ValueError(
                        "Missing or duplicate fixed-cutoff reference metrics"
                    )
                for name in (
                    "precision",
                    "recall",
                    "f1",
                    "mcc",
                    "pr_auc",
                    "accuracy",
                    "tp",
                    "fp",
                    "tn",
                    "fn",
                ):
                    expected = prior.iloc[0][name]
                    if fixed[name] is None:
                        if not pd.isna(expected):
                            raise ValueError("Undefined metric differs from source")
                    else:
                        np.testing.assert_allclose(
                            fixed[name], expected, rtol=0, atol=1e-12
                        )
                if fixed["pr_auc"] != chosen["pr_auc"]:
                    raise ValueError(
                        "Changing the cutoff unexpectedly changed Average Precision"
                    )
                fixed_rows.append(fixed)
                chosen_rows.append(chosen)
                fixed_pair[kind], selected_pair[kind] = fixed, chosen
                selected_predictions.rename(
                    columns={f"{kind}_predicted_label": f"{kind}_label_at_43"},
                    inplace=True,
                )
                selected_predictions[f"{kind}_selected_label"] = (
                    scores[kind] >= threshold
                ).astype("uint8")
                sweep = pd.read_parquet(
                    source / candidate / f"{kind}_time_sweep.parquet"
                )
                extremes = sweep.groupby("source_row_number").score.agg(["min", "max"])
                boundary.append(
                    {
                        "candidate": candidate,
                        "model": kind,
                        "rows": len(extremes),
                        "selected_threshold": threshold,
                        "crosses_at_43": float(
                            (
                                (extremes["min"] < fixed_cutoff)
                                & (extremes["max"] >= fixed_cutoff)
                            ).mean()
                        ),
                        "crosses_at_selected": float(
                            (
                                (extremes["min"] < threshold)
                                & (extremes["max"] >= threshold)
                            ).mean()
                        ),
                        "mean_range_pp_unchanged": float(
                            100 * (extremes["max"] - extremes["min"]).mean()
                        ),
                    }
                )
            selected_predictions.to_parquet(
                folder / "predictions_selected.parquet", index=False
            )
            mean_fixed = sum(fixed_pair[k]["f1"] for k in MODELS) / 2
            mean_chosen = sum(selected_pair[k]["f1"] for k in MODELS) / 2
            if mean_chosen + 1e-15 < mean_fixed:
                raise ValueError(
                    "Best shared mean F1 cannot be worse than the included 43% cutoff"
                )
            summary.append(
                {
                    "candidate": candidate,
                    "max_depth": item["max_depth"],
                    "min_samples_leaf": item["min_samples_leaf"],
                    "threshold": threshold,
                    "rf_f1_fixed": fixed_pair["rf"]["f1"],
                    "rf_smote_f1_fixed": fixed_pair["rf_smote"]["f1"],
                    "rf_f1_selected": selected_pair["rf"]["f1"],
                    "rf_smote_f1_selected": selected_pair["rf_smote"]["f1"],
                    "mean_f1_fixed": mean_fixed,
                    "mean_f1_selected": mean_chosen,
                    "delta_mean_f1": mean_chosen - mean_fixed,
                    "mean_f1_fraction": decision["mean_f1_fraction"],
                }
            )
            manifest["completed"].append(candidate)
            write_json(output / "manifest.json", manifest)
        fixed, selected, pairs = (
            pd.DataFrame(fixed_rows),
            pd.DataFrame(chosen_rows),
            pd.DataFrame(summary),
        )
        changes = selected.copy()
        for name in (
            "precision",
            "recall",
            "f1",
            "mcc",
            "pr_auc",
            "tp",
            "fp",
            "tn",
            "fn",
        ):
            changes[f"delta_{name}_vs_43"] = selected[name] - fixed[name]
        for name, table in {
            "metrics_fixed_43": fixed,
            "metrics_selected": selected,
            "pair_summary": pairs,
            "changes_vs_43": changes,
            "threshold_grid": pd.concat(all_grids, ignore_index=True),
            "time_boundary_changes": pd.DataFrame(boundary),
        }.items():
            save_table(table, output, name)
        write_json(
            output / "verification.json",
            {
                "status": "passed",
                "pairs": checks,
                "aligned_validation_rows": len(labels),
                "actual_fraud": int(labels.sum()),
                "thresholds_verified": 100 * len(checks),
                "source_fixed_metrics_reproduced": True,
                "average_precision_unchanged": True,
                "new_fits": 0,
                "test_used": False,
            },
        )
        render(output, source, pairs, selected)
        verify_hashes(inventory)
        verify_hashes(implementation)
        manifest.update(
            status="complete",
            source_preservation_verified=True,
            files={
                p.relative_to(output).as_posix(): file_sha256(p)
                for p in output.rglob("*")
                if p.is_file() and p.name != "manifest.json"
            },
        )
        write_json(output / "manifest.json", manifest)
        return output
    except BaseException as exc:
        manifest.update(
            status="failed", failure_type=type(exc).__name__, error=str(exc)
        )
        write_json(output / "manifest.json", manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "outputs" / "forest_regularization_20261117",
    )
    parser.add_argument("--run-id")
    args = parser.parse_args()
    output = run(args.source, ROOT / "outputs", args.run_id)
    print(json.dumps({"success": True, "output": str(output), "new_fits": 0}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
