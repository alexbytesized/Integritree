"""Descriptive validation comparison of 1:1 and 1:3 RF-SMOTE reports."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from integritree.config import ExperimentConfig
from integritree.ml.data import file_sha256, write_json
from integritree.ml.evaluation import METRICS, metrics
from integritree.ml.research import load_report, new_run
from integritree.ml.selection import THRESHOLDS
from integritree.settings import load_settings

MODEL_LABELS = {
    "rf": "Benchmark RF",
    "rf_smote_1_to_1": "RF-SMOTE 1:1",
    "rf_smote_1_to_3": "RF-SMOTE 1:3",
}
REQUIRED_FILES = {
    "SUMMARY.md", "score_metrics.csv", "threshold_metrics.csv", "ratio_differences.csv",
    "figures/precision_recall.svg", "figures/threshold_tradeoffs.svg",
}


def _evaluation_configuration(report: Path) -> tuple[dict, ExperimentConfig]:
    value = json.loads((report / "evaluation_configuration.json").read_text(encoding="utf-8"))
    requested = ExperimentConfig.model_validate(value["requested_configuration"])
    smote = value.get("effective_smote", requested.smote.model_dump())
    return value | {"effective_smote": smote}, requested


def _metric_value(result: dict, name: str):
    return result["metrics"][name]["value"]


def _save_figures(labels, scores: dict[str, np.ndarray], threshold_rows: pd.DataFrame,
                  output: Path) -> None:
    import os
    os.environ.setdefault("MPLCONFIGDIR", str(load_settings().backend_root / "runtime/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    output.mkdir()
    fig, ax = plt.subplots()
    for key, values in scores.items():
        precision, recall, _ = precision_recall_curve(labels, values)
        ax.step(recall, precision, where="post", label=MODEL_LABELS[key])
    ax.set(xlabel="Recall", ylabel="Precision",
           title="Validation precision-recall curves; summary metric: Average Precision")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "precision_recall.svg")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for ax, metric_name in zip(axes.flat, ("precision", "recall", "f1", "mcc")):
        for key in scores:
            selected = threshold_rows[threshold_rows.model == key]
            ax.plot(selected.threshold, selected[metric_name], marker="o", markersize=2,
                    label=MODEL_LABELS[key])
        ax.set(title=metric_name.upper() if metric_name != "mcc" else "MCC",
               ylabel=metric_name, xlabel="Threshold")
        ax.grid(alpha=.25)
    axes.flat[0].legend(fontsize="small")
    fig.suptitle("Validation metrics across the common threshold grid")
    fig.tight_layout()
    fig.savefig(output / "threshold_tradeoffs.svg")
    plt.close(fig)


def _summary(rows: int, fraud: int, score_rows: pd.DataFrame,
             threshold_rows: pd.DataFrame) -> str:
    ap = score_rows.set_index("model")["pr_auc"]
    lines = [
        "# RF-SMOTE ratio validation comparison",
        "",
        "**Validation only. The held-out test split was not used.**",
        "",
        f"Records: {rows:,}; fraud labels: {fraud:,}.",
        "Random Forest setting: 200 trees, maximum depth 20.",
        "The 1:3 branch was added after review of the 1:1 validation results, and this",
        "tree setting was inherited from the earlier 1:1 search. The study is descriptive:",
        "it does not select a sampling ratio, classification cutoff, or final model.",
        "",
        "## Score-based metric",
        "",
        "| Model | PR-AUC (Average Precision) |",
        "| --- | ---: |",
    ]
    for key in MODEL_LABELS:
        lines.append(f"| {MODEL_LABELS[key]} | {ap[key]:.6f} |")
    lines += [
        "",
        "## Reference checkpoints",
        "",
        "The complete 0.05-0.95 grid is in `threshold_metrics.csv`; these two rows are",
        "included only as convenient checkpoints and are not selected cutoffs.",
        "",
        "| Threshold | Model | Precision | Recall | F1 | MCC | Accuracy |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    checkpoints = threshold_rows[threshold_rows.threshold.isin([.5, .95])]
    for row in checkpoints.itertuples(index=False):
        lines.append(
            f"| {row.threshold:.2f} | {MODEL_LABELS[row.model]} | {row.precision:.6f} | "
            f"{row.recall:.6f} | {row.f1:.6f} | {row.mcc:.6f} | {row.accuracy:.6f} |")
    lines += [
        "",
        "`ratio_differences.csv` reports RF-SMOTE 1:3 minus RF-SMOTE 1:1 in raw",
        "metric units at each threshold. These differences are descriptive and are not",
        "a significance test. No test evaluation or SHAP computation is part of this report.",
        "",
    ]
    return "\n".join(lines)


def compare_ratio_reports(one_to_one_report: Path, one_to_three_report: Path,
                          output_root: Path, run_id: str | None = None,
                          progress=print) -> Path:
    reports = {"one_to_one": one_to_one_report.resolve(),
               "one_to_three": one_to_three_report.resolve()}
    loaded = {key: load_report(path) for key, path in reports.items()}
    for key, (metadata, _) in loaded.items():
        if metadata.get("split") != "validation" or metadata.get("official_test"):
            raise ValueError(f"{key} must be a validation-only report")
    first_meta, first = loaded["one_to_one"]
    third_meta, third = loaded["one_to_three"]
    if first_meta["prepared_metadata_sha256"] != third_meta["prepared_metadata_sha256"]:
        raise ValueError("Ratio reports use different prepared data")
    identity = ["transaction_id", "source_row_number", "actual_label", "split"]
    if not first[identity].equals(third[identity]):
        raise ValueError("Ratio reports have different validation identities or labels")
    if not np.array_equal(first.rf_risk_score.to_numpy(), third.rf_risk_score.to_numpy()):
        raise ValueError("Benchmark RF scores differ between ratio reports")

    first_effective, first_config = _evaluation_configuration(reports["one_to_one"])
    third_effective, third_config = _evaluation_configuration(reports["one_to_three"])
    first_ratio = first_effective["effective_smote"]["sampling_ratio"]
    third_ratio = third_effective["effective_smote"]["sampling_ratio"]
    if not math.isclose(first_ratio, 1.0) or not math.isclose(third_ratio, 1 / 3):
        raise ValueError("Reports must represent SMOTE ratios 1:1 and 1:3")
    for effective in (first_effective, third_effective):
        rf = effective["effective_random_forest"]
        if rf["n_estimators"] != 200 or rf["max_depth"] != 20:
            raise ValueError("Ratio comparison requires 200 trees and maximum depth 20")
    if first_config.evaluation != third_config.evaluation:
        raise ValueError("Ratio reports use different evaluation policies")

    labels = first.actual_label.to_numpy()
    scores = {
        "rf": first.rf_risk_score.to_numpy(),
        "rf_smote_1_to_1": first.rf_smote_risk_score.to_numpy(),
        "rf_smote_1_to_3": third.rf_smote_risk_score.to_numpy(),
    }
    score_rows = []
    threshold_rows = []
    evaluated: dict[tuple[float, str], dict] = {}
    for key, values in scores.items():
        result = metrics(labels, values, .5, first_config.evaluation)
        score_rows.append({"model": key,
                           "sampling_ratio": None if key == "rf" else (1.0 if key.endswith("1_to_1") else 1 / 3),
                           "pr_auc": _metric_value(result, "pr_auc")})
    for threshold in THRESHOLDS:
        for key, values in scores.items():
            result = metrics(labels, values, threshold, first_config.evaluation)
            evaluated[(threshold, key)] = result
            matrix = result["confusion_matrix"]
            threshold_rows.append({
                "threshold": threshold, "model": key,
                "sampling_ratio": None if key == "rf" else (1.0 if key.endswith("1_to_1") else 1 / 3),
                "tn": matrix["tn"], "fp": matrix["fp"], "fn": matrix["fn"], "tp": matrix["tp"],
                **{name: _metric_value(result, name) for name in METRICS if name != "pr_auc"},
            })
    difference_rows = []
    for threshold in THRESHOLDS:
        old = evaluated[(threshold, "rf_smote_1_to_1")]
        new = evaluated[(threshold, "rf_smote_1_to_3")]
        row = {"threshold": threshold}
        for name in METRICS:
            old_value = _metric_value(old, name)
            new_value = _metric_value(new, name)
            row[f"{name}_1_to_1"] = old_value
            row[f"{name}_1_to_3"] = new_value
            row[f"{name}_difference"] = None if old_value is None or new_value is None else new_value - old_value
        difference_rows.append(row)

    score_frame = pd.DataFrame(score_rows)
    threshold_frame = pd.DataFrame(threshold_rows)
    difference_frame = pd.DataFrame(difference_rows)
    output = new_run(output_root, "ratio_comparison", run_id)
    manifest = {
        "schema_version": 1, "kind": "smote_ratio_validation_comparison", "status": "running",
        "run_id": output.name, "created_at": datetime.now(timezone.utc).isoformat(),
        "split": "validation", "test_used": False, "selection_performed": False,
        "thresholds": list(THRESHOLDS), "rows": len(labels), "actual_fraud": int(labels.sum()),
        "reports": {key: {"path": str(path), "metadata_sha256": file_sha256(path / "metadata.json")}
                    for key, path in reports.items()},
    }
    write_json(output / "metadata.json", manifest)
    try:
        score_frame.to_csv(output / "score_metrics.csv", index=False, float_format="%.17g")
        threshold_frame.to_csv(output / "threshold_metrics.csv", index=False, float_format="%.17g")
        difference_frame.to_csv(output / "ratio_differences.csv", index=False, float_format="%.17g")
        (output / "SUMMARY.md").write_text(
            _summary(len(labels), int(labels.sum()), score_frame, threshold_frame), encoding="utf-8")
        _save_figures(labels, scores, threshold_frame, output / "figures")
        manifest["files"] = {name: file_sha256(output / name) for name in sorted(REQUIRED_FILES)}
        manifest["status"] = "complete"
        write_json(output / "metadata.json", manifest)
        load_ratio_comparison(output)
        progress(f"Saved validation-only ratio comparison to {output}")
        return output
    except BaseException as exc:
        manifest.update(status="failed", failure_type=type(exc).__name__)
        write_json(output / "metadata.json", manifest)
        raise


def load_ratio_comparison(path: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    if (metadata.get("schema_version") != 1
            or metadata.get("kind") != "smote_ratio_validation_comparison"
            or metadata.get("status") != "complete"
            or metadata.get("split") != "validation"
            or metadata.get("test_used") is not False
            or metadata.get("selection_performed") is not False
            or set(metadata.get("files", {})) != REQUIRED_FILES):
        raise ValueError("Ratio comparison report is incomplete or unsupported")
    for name, digest in metadata["files"].items():
        target = (path / name).resolve()
        if not target.is_relative_to(path.resolve()) or file_sha256(target) != digest:
            raise ValueError(f"Ratio comparison fingerprint mismatch: {name}")
    scores = pd.read_csv(path / "score_metrics.csv")
    thresholds = pd.read_csv(path / "threshold_metrics.csv")
    differences = pd.read_csv(path / "ratio_differences.csv")
    observed_thresholds = np.sort(thresholds["threshold"].unique())
    if (len(scores) != 3 or len(thresholds) != len(THRESHOLDS) * 3
            or len(differences) != len(THRESHOLDS)
            or set(thresholds["model"]) != set(MODEL_LABELS)
            or observed_thresholds.shape != (len(THRESHOLDS),)
            or not np.allclose(observed_thresholds, THRESHOLDS, rtol=0, atol=1e-15)):
        raise ValueError("Ratio comparison tables do not match the declared threshold grid")
    return metadata, scores, thresholds, differences


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--one-to-one-report", type=Path, required=True)
    parser.add_argument("--one-to-three-report", type=Path, required=True)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    try:
        settings = load_settings()
        resolve = lambda p: p if p.is_absolute() else settings.backend_root / p
        output = compare_ratio_reports(resolve(args.one_to_one_report),
                                       resolve(args.one_to_three_report),
                                       settings.reports_dir, args.run_id,
                                       lambda message: print(message, file=sys.stderr, flush=True))
    except (ValueError, OSError) as exc:
        print(json.dumps({"success": False, "error": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps({"success": True, "report": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
