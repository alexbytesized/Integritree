"""Integrity-checked evaluation reports; validation by default, frozen selection required for test."""

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import sys
import uuid
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve
from integritree.config import load_experiment
from integritree.ml.artifacts import load_bundle
from integritree.ml.data import file_sha256, write_json
from integritree.ml.evaluation import evaluate_pair, validate_policy
from integritree.ml.inference import predict_features
from integritree.ml.preparation import load_prepared_split
from integritree.settings import load_settings


def new_run(root, prefix, run_id=None):
    name = (
        run_id
        or prefix
        + "_"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        + "_"
        + uuid.uuid4().hex[:8]
    )
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,100}", name):
        raise ValueError("Invalid run ID")
    path = root.resolve() / name
    path.mkdir(parents=True, exist_ok=False)
    return path


def verify_prepared(bundle, prepared):
    metadata = json.loads((prepared / "metadata.json").read_text())
    if metadata.get("status") != "complete":
        raise ValueError("Preparation bundle is incomplete")
    if (
        file_sha256(prepared / "metadata.json")
        != bundle.metadata["prepared_metadata_sha256"]
    ):
        raise ValueError("Prepared provenance differs from the trained model")
    for name in ("preprocessing.json", "split_manifest.parquet"):
        if file_sha256(prepared / name) != metadata["files"][name]:
            raise ValueError(f"Preparation fingerprint mismatch: {name}")
    return metadata


def paired_scores(bundle, features, identities, progress=lambda _: None):
    frames = []
    for start in range(0, len(features), 25000):
        stop = start + 25000
        frames.append(
            predict_features(
                bundle.models,
                features.iloc[start:stop],
                identities[start:stop],
                bundle.config.scoring.threshold,
                bundle.metadata["run_id"],
            )
        )
        progress(
            f"Predicted {min(stop, len(features)):,}/{len(features):,} paired records"
        )
    return pd.concat(frames, ignore_index=True)


def save_figures(predictions, results, output):
    os.environ.setdefault(
        "MPLCONFIGDIR", str(load_settings().backend_root / "runtime/matplotlib")
    )
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    output.mkdir()
    fig, ax = plt.subplots()
    for model in ("rf", "rf_smote"):
        if results["models"][model]["actual_fraud"]:
            p, r, _ = precision_recall_curve(
                predictions.actual_label, predictions[f"{model}_risk_score"]
            )
            # AP uses a stepwise convention, not trapezoidal interpolation.
            ax.step(r, p, where="post", label=model)
    ax.set(
        xlabel="Recall",
        ylabel="Precision",
        title="Precision-recall curves; summary: Average Precision",
    )
    if results["models"]["rf"]["actual_fraud"]:
        ax.legend()
    fig.tight_layout()
    fig.savefig(output / "precision_recall.svg")
    plt.close(fig)
    for model in ("rf", "rf_smote"):
        matrix = np.array(results["models"][model]["confusion_matrix"]["matrix"])
        fig, ax = plt.subplots()
        ax.imshow(matrix, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center")
        ax.set(
            xticks=[0, 1],
            yticks=[0, 1],
            xticklabels=["Legitimate", "Fraud"],
            yticklabels=["Legitimate", "Fraud"],
            xlabel="Predicted",
            ylabel="Actual",
            title=model,
        )
        fig.tight_layout()
        fig.savefig(output / f"{model}_confusion.svg")
        plt.close(fig)


def evaluate_run(
    model_path,
    prepared,
    config,
    output_root,
    split="validation",
    run_id=None,
    progress=print,
):
    config.require_ready("evaluate")
    validate_policy(config.evaluation)
    if split not in ("validation", "test"):
        raise ValueError("Research evaluation supports validation or final test only")
    bundle = load_bundle(model_path)
    if split == "test" and bundle.metadata.get("stage") != "validation_selected":
        raise ValueError(
            "Official test evaluation requires a frozen validation-selected bundle"
        )
    # Runtime evaluation policy is separate from immutable model configuration.
    if (
        config.dataset != bundle.config.dataset
        or config.preprocessing != bundle.config.preprocessing
    ):
        raise ValueError("Evaluation dataset/features differ from model configuration")
    preparation = verify_prepared(bundle, prepared)
    output = new_run(output_root, "evaluate", run_id)
    manifest = {
        "schema_version": 1,
        "status": "running",
        "run_id": output.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "split": split,
        "official_test": split == "test",
        "model_run_id": bundle.metadata["run_id"],
        "model_metadata_sha256": file_sha256(model_path / "metadata.json"),
        "prepared_metadata_sha256": file_sha256(prepared / "metadata.json"),
        "dataset_sha256": bundle.config.dataset.sha256,
        "policy": config.evaluation.model_dump(),
        "package_versions": {
            p: version(p) for p in ("scikit-learn", "numpy", "scipy", "matplotlib")
        },
        "implementation_sha256": {
            p.name: file_sha256(p)
            for p in (Path(__file__), Path(__file__).with_name("evaluation.py"))
        },
    }
    write_json(output / "metadata.json", manifest)
    try:
        features, labels, rows = load_prepared_split(prepared, split)
        ids = [f"{bundle.config.dataset.sha256}:{i}" for i in rows]
        predictions = paired_scores(bundle, features, ids, progress)
        predictions.insert(1, "source_row_number", rows.to_numpy())
        predictions.insert(2, "actual_label", labels.to_numpy())
        predictions.insert(3, "split", split)
        predictions["threshold"] = bundle.config.scoring.threshold
        results = evaluate_pair(
            labels,
            predictions.rf_risk_score,
            predictions.rf_smote_risk_score,
            bundle.config.scoring.threshold,
            config.evaluation,
        )
        predictions.to_parquet(output / "predictions.parquet", index=False)
        predictions.to_csv(
            output / "predictions.csv", index=False, float_format="%.17g"
        )
        write_json(
            output / "evaluation_configuration.json",
            {
                "evaluation": config.evaluation.model_dump(),
                "effective_scoring": bundle.config.scoring.model_dump(),
                "effective_random_forest": bundle.config.random_forest.model_dump(),
                "effective_smote": bundle.config.smote.model_dump(),
                "model_metadata_sha256": manifest["model_metadata_sha256"],
                "requested_configuration": config.model_dump(),
                "note": "Effective model settings come from the saved bundle, not the requested configuration.",
            },
        )
        write_json(
            output / "metrics.json", {"split": split, "models": results["models"]}
        )
        write_json(output / "statistical_tests.json", results["statistical_test"])
        write_json(output / "comparisons.json", results["descriptive_comparisons"])
        save_figures(predictions, results, output / "figures")
        manifest.update(
            status="complete",
            rows=len(predictions),
            threshold=bundle.config.scoring.threshold,
            split_manifest_sha256=preparation["split_manifest_sha256"],
        )
        manifest["files"] = {
            p.relative_to(output).as_posix(): file_sha256(p)
            for p in output.rglob("*")
            if p.is_file() and p.name != "metadata.json"
        }
        write_json(output / "metadata.json", manifest)
        return output
    except BaseException as exc:
        manifest.update(status="failed", failure_type=type(exc).__name__)
        write_json(output / "metadata.json", manifest)
        raise


def load_report(path):
    metadata = json.loads((path / "metadata.json").read_text())
    if metadata.get("status") != "complete" or metadata.get("schema_version") != 1:
        raise ValueError("Evaluation report is incomplete or unsupported")
    required = {
        "predictions.parquet",
        "predictions.csv",
        "metrics.json",
        "statistical_tests.json",
        "comparisons.json",
        "evaluation_configuration.json",
        "figures/precision_recall.svg",
        "figures/rf_confusion.svg",
        "figures/rf_smote_confusion.svg",
    }
    if set(metadata.get("files", {})) != required:
        raise ValueError("Evaluation report file manifest is incomplete")
    for name, digest in metadata["files"].items():
        target = (path / name).resolve()
        if not target.is_relative_to(path.resolve()) or file_sha256(target) != digest:
            raise ValueError("Evaluation report fingerprint mismatch")
    predictions = pd.read_parquet(path / "predictions.parquet")
    if len(predictions) != metadata["rows"] or not predictions.transaction_id.is_unique:
        raise ValueError("Invalid evaluation prediction identities")
    return metadata, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    try:
        settings = load_settings()
        resolve = lambda p: p if p.is_absolute() else settings.backend_root / p
        config = load_experiment(resolve(args.config or settings.experiment_config))
        output = evaluate_run(
            resolve(args.models),
            resolve(args.prepared),
            config,
            settings.reports_dir,
            args.split,
            args.run_id,
            lambda x: print(x, file=sys.stderr, flush=True),
        )
    except (ValueError, OSError, MemoryError) as exc:
        print(json.dumps({"success": False, "error": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps({"success": True, "report": str(output)}))
    return 0
