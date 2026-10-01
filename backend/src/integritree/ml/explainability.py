"""Shared probability-space TreeSHAP with reproducible background, provenance, and coverage."""

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from integritree.config import load_experiment
from integritree.ml.artifacts import load_bundle
from integritree.ml.data import file_sha256, write_json
from integritree.ml.features import FEATURE_COLUMNS
from integritree.ml.inference import feature_matrix, fraud_scores
from integritree.ml.research import new_run, verify_prepared, load_report
from integritree.ml.shap_adapter import forest_representation
from integritree.settings import load_settings


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def validate_shap_policy(policy):
    expected = {
        "output_space": "probability",
        "background_strategy": "uniform_original_training_without_replacement",
        "perturbation": "interventional",
        "explanation_coverage": "on_demand_and_sampled_global_explicit_full_job",
    }
    for name, value in expected.items():
        if getattr(policy, name) != value:
            raise ValueError(f"Unsupported or unresolved shap.{name}")
    if any(
        getattr(policy, n) is None
        for n in (
            "seed",
            "background_size",
            "global_sample_size",
            "additivity_tolerance",
            "positive_tolerance",
        )
    ):
        raise ValueError("SHAP numerical/sampling settings are unresolved")


def sample_feature_rows(path, count, seed):
    """Uniform seeded sample without materializing the full training matrix."""
    source = pq.ParquetFile(path)
    total = source.metadata.num_rows
    count = min(count, total)
    positions = np.sort(
        np.random.default_rng(seed).choice(total, size=count, replace=False)
    )
    collected, offset = [], 0
    for batch in source.iter_batches(batch_size=25000):
        chosen = (
            positions[(positions >= offset) & (positions < offset + len(batch))]
            - offset
        )
        if len(chosen):
            collected.append(batch.to_pandas().iloc[chosen])
        offset += len(batch)
        if offset > positions[-1]:
            break
    return pd.concat(collected, ignore_index=True)


def create_background(bundle, model_path, prepared, output_root, policy):
    validate_shap_policy(policy)
    metadata = verify_prepared(bundle, prepared)
    train_path = prepared / "train_features.parquet"
    if file_sha256(train_path) != metadata["files"]["train_features.parquet"]:
        raise ValueError("Training feature fingerprint mismatch")
    settings = {
        "prepared_metadata_sha256": file_sha256(prepared / "metadata.json"),
        "training_features_sha256": metadata["files"]["train_features.parquet"],
        "strategy": policy.background_strategy,
        "size": policy.background_size,
        "seed": policy.seed,
    }
    folder = output_root / ("background_" + digest(settings))
    if folder.exists():
        manifest = json.loads((folder / "metadata.json").read_text())
        if manifest.get("status") != "complete" or manifest.get("settings") != settings:
            raise ValueError("Background cache is incomplete or incompatible")
        if file_sha256(folder / "features.parquet") != manifest["features_sha256"]:
            raise ValueError("Background fingerprint mismatch")
        frame = pd.read_parquet(folder / "features.parquet")
        return folder, frame
    folder.mkdir(parents=True, exist_ok=False)
    write_json(folder / "metadata.json", {"status": "running", "settings": settings})
    frame = sample_feature_rows(train_path, policy.background_size, policy.seed)
    frame.to_parquet(folder / "features.parquet", index=False)
    manifest = {
        "status": "complete",
        "settings": settings,
        "rows": len(frame),
        "source_row_numbers": frame.source_row_number.tolist(),
        "dataset_sha256": bundle.config.dataset.sha256,
        "features_sha256": file_sha256(folder / "features.parquet"),
        "reference_distribution": "original training; no class balancing; no synthetic/held-out rows",
    }
    write_json(folder / "metadata.json", manifest)
    return folder, frame


def readable_features(values, preprocessor):
    original = dict(zip(FEATURE_COLUMNS, map(float, values)))
    state = preprocessor.state
    for name, scale, offset in zip(state.scaled_columns, state.scale, state.offset):
        original[name] = (original[name] - offset) / scale
    rendered = []
    unknown_type = all(
        original[n] == 0 for n in FEATURE_COLUMNS if n.startswith("type_")
    )
    for name in FEATURE_COLUMNS:
        value = original[name]
        if name.startswith("type_"):
            kind = name[5:]
            text = (
                f"Transaction type unknown; {kind} indicator = 0"
                if unknown_type
                else f"Transaction type is {kind}"
                if value == 1
                else f"Transaction type is not {kind}"
                if value == 0
                else f"{kind} indicator = {value:.6g}"
            )
        elif name == "log_amount":
            text = (
                f"Transaction amount = {np.expm1(value):.6g} (log-transformed feature)"
            )
        elif name == "hour_of_day":
            text = f"Simulation hour = {value:.6g}"
        elif name == "day_of_week":
            text = f"Simulation day index = {value:.6g}"
        elif name == "is_zero_amount":
            text = (
                "Amount is zero"
                if value == 1
                else "Amount is not zero"
                if value == 0
                else f"Zero-amount indicator = {value:.6g}"
            )
        else:
            entity = "Origin" if name == "is_merchant_origin" else "Destination"
            text = (
                f"{entity} is a merchant"
                if value == 1
                else f"{entity} is not a merchant"
                if value == 0
                else f"{entity} merchant indicator = {value:.6g}"
            )
        rendered.append(text)
    return rendered


def summarize(values, readable, tolerance):
    positive = sorted(
        (i for i, v in enumerate(values) if v > tolerance),
        key=lambda i: (-values[i], i),
    )
    negative = sorted(
        (i for i, v in enumerate(values) if v < -tolerance),
        key=lambda i: (values[i], i),
    )
    top = (
        {
            "status": "available",
            "feature": FEATURE_COLUMNS[positive[0]],
            "contribution": float(values[positive[0]]),
        }
        if positive
        else {
            "status": "no_positive_contributor",
            "feature": None,
            "contribution": None,
        }
    )
    parts = []
    if positive:
        parts.append(
            "Factors increasing this model's fraud score: "
            + "; ".join(
                f"{readable[i]} (+{values[i] * 100:.4g} percentage points)"
                for i in positive[:2]
            )
            + "."
        )
    else:
        parts.append("No risk-increasing contributor above the numerical tolerance.")
    if negative:
        parts.append(
            "Factors decreasing this model's fraud score: "
            + "; ".join(
                f"{readable[i]} ({values[i] * 100:.4g} percentage points)"
                for i in negative[:2]
            )
            + "."
        )
    parts.append(
        "These are contributions to the model output, not proven causes of fraud. The chart includes all features."
    )
    return top, " ".join(parts)


_WATERFALL_LOCK = threading.Lock()


def waterfall(explanation, output, *, display_label=None):
    # HTTP display rendering and the explanation worker share Matplotlib state.
    with _WATERFALL_LOCK:
        _render_waterfall(explanation, output, display_label=display_label)


def _render_waterfall(explanation, output, *, display_label=None):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    values = [f["contribution"] * 100 for f in explanation["features"]]
    names = [f["readable_value"] for f in explanation["features"]]
    # Show all eleven effects; order by magnitude, with deterministic ties.
    order = sorted(range(len(values)), key=lambda i: (-abs(values[i]), i))
    fig, ax = plt.subplots(figsize=(10, 7))
    start = explanation["base_value"] * 100
    endpoints = [start, explanation["output_value"] * 100]
    for row, i in enumerate(order):
        end = start + values[i]
        ax.barh(
            row,
            abs(values[i]),
            left=min(start, end),
            color="#c83e4d" if values[i] > 0 else "#2874a6",
        )
        ax.plot([end, end], [row - 0.4, row + 0.4], color="black", linewidth=0.5)
        ax.text(max(start, end), row, f" {values[i]:+.3g} pp", va="center", fontsize=8)
        start = end
        endpoints.append(end)
    ax.set_yticks(range(len(order)), [names[i] for i in order])
    ax.invert_yaxis()
    ax.axvline(
        explanation["base_value"] * 100, color="gray", linestyle="--", label="Reference"
    )
    ax.axvline(
        explanation["output_value"] * 100,
        color="black",
        linestyle=":",
        label="Prediction",
    )
    span = max(max(endpoints) - min(endpoints), 1.0)
    ax.set_xlim(min(endpoints) - 0.08 * span, max(endpoints) + 0.25 * span)
    ax.set_xlabel("Model fraud score (%) — contributions in percentage points")
    record_label = (
        explanation["transaction_id"] if display_label is None else display_label
    )
    if len(record_label) > 35:
        record_label = record_label[:12] + "..." + record_label[-16:]
    model_label = (
        explanation["model"]
        if display_label is None
        else {"rf": "Benchmark RF", "rf_smote": "RF-SMOTE"}[explanation["model"]]
    )
    precision = ".4g" if display_label is None else ".2f"
    ax.set_title(
        f"{model_label} | {record_label}\n"
        f"Reference {explanation['base_value'] * 100:{precision}}% → output {explanation['output_value'] * 100:{precision}}%"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, format=Path(output).suffix.lstrip("."))
    plt.close(fig)


class ExplanationEngine:
    """Cache key includes model/input/reference/policy/code/dependency identity."""

    def __init__(self, model_path, prepared, policy, cache_root, *, bundle=None):
        os.environ.setdefault(
            "MPLCONFIGDIR", str(load_settings().backend_root / "runtime/matplotlib")
        )
        import shap

        validate_shap_policy(policy)
        self.model_path = Path(model_path)
        self.bundle = bundle if bundle is not None else load_bundle(self.model_path)
        self.policy = policy
        self.cache_root = Path(cache_root)
        self.background_path, background = create_background(
            self.bundle, self.model_path, Path(prepared), self.cache_root, policy
        )
        self.background = feature_matrix(background.loc[:, FEATURE_COLUMNS]).astype(
            np.float32
        )
        self.identity = {
            "model_metadata_sha256": file_sha256(self.model_path / "metadata.json"),
            "background_metadata_sha256": file_sha256(
                self.background_path / "metadata.json"
            ),
            "policy": policy.model_dump(),
            "shap_version": version("shap"),
            "implementation_sha256": file_sha256(Path(__file__)),
            "adapter_sha256": file_sha256(Path(__file__).with_name("shap_adapter.py")),
            "representation": "float32_predicate_equivalent_gated_subtree_sum",
        }
        self.explainers = {}
        for name, model in self.bundle.models.items():
            masker = shap.maskers.Independent(
                self.background, max_samples=len(self.background)
            )
            self.explainers[name] = shap.TreeExplainer(
                forest_representation(model),
                masker,
                model_output="probability",
                feature_perturbation="interventional",
            )
            if len(self.explainers[name].data) != len(self.background):
                raise ValueError("SHAP unexpectedly changed the reference sample size")
            np.testing.assert_allclose(
                self.explainers[name].model.predict(self.background),
                model.predict_proba(self.background),
                rtol=0,
                atol=1e-12,
                err_msg="SHAP representation changed reference predictions",
            )

    def explain(self, features, transaction_ids, analysis_id="local"):
        values = feature_matrix(features)
        ids = list(transaction_ids)
        if (
            len(ids) != len(values)
            or len(set(ids)) != len(ids)
            or any(not isinstance(i, str) or not i for i in ids)
        ):
            raise ValueError("SHAP requires unique aligned nonempty transaction IDs")
        outputs = []
        for row, (record_id, record) in enumerate(zip(ids, values)):
            readable = readable_features(record, self.bundle.preprocessor)
            for name, model in self.bundle.models.items():
                key = digest(
                    self.identity
                    | {
                        "analysis_id": analysis_id,
                        "record_id": record_id,
                        "model": name,
                        "features": record.tolist(),
                    }
                )
                folder = self.cache_root / "records" / key
                if (folder / "manifest.json").exists():
                    manifest = json.loads((folder / "manifest.json").read_text())
                    if manifest.get("status") != "complete" or any(
                        file_sha256(folder / n) != h
                        for n, h in manifest["files"].items()
                    ):
                        raise ValueError("Explanation cache fingerprint mismatch")
                    outputs.append(
                        json.loads((folder / "explanation.json").read_text())
                    )
                    continue
                folder.mkdir(parents=True, exist_ok=True)
                started = time.monotonic()
                explanation_input = record.reshape(1, -1).astype(np.float32)
                result = self.explainers[name](
                    explanation_input, check_additivity=False
                )
                fraud_column = int(np.flatnonzero(model.classes_ == 1)[0])
                if result.values.shape != (1, len(FEATURE_COLUMNS), 2):
                    raise ValueError("Unsupported SHAP class/feature output shape")
                contributions = np.asarray(
                    result.values[0, :, fraud_column], dtype=float
                )
                base = float(result.base_values[0, fraud_column])
                score = float(fraud_scores(model, record.reshape(1, -1))[0])
                difference = abs(base + float(contributions.sum()) - score)
                # Check reconstruction against saved-model probability, not adapter output.
                if (
                    not np.isfinite(contributions).all()
                    or not np.isfinite(base)
                    or difference > self.policy.additivity_tolerance
                ):
                    raise ValueError(
                        "SHAP contributions do not reconstruct the model probability"
                    )
                top, narrative = summarize(
                    contributions, readable, self.policy.positive_tolerance
                )
                predicted = int(score >= self.bundle.config.scoring.threshold)
                narrative += f" With the common cutoff {self.bundle.config.scoring.threshold:.2f}, this model predicted {'fraud' if predicted else 'legitimate'}."
                item = {
                    "status": "computed",
                    "transaction_id": record_id,
                    "analysis_id": analysis_id,
                    "model": name,
                    "model_run_id": self.bundle.metadata["run_id"],
                    "output_space": "fraud_probability",
                    "base_value": base,
                    "output_value": score,
                    "predicted_label": predicted,
                    "threshold": self.bundle.config.scoring.threshold,
                    "features": [
                        {
                            "feature": f,
                            "model_value": float(record[i]),
                            "readable_value": readable[i],
                            "contribution": float(contributions[i]),
                        }
                        for i, f in enumerate(FEATURE_COLUMNS)
                    ],
                    "top_positive_contributor": top,
                    "narrative": narrative,
                    "chart_description": f"{name} waterfall from {base:.8g} to {score:.8g}. "
                    + narrative,
                    "additivity_error": difference,
                    "elapsed_seconds": time.monotonic() - started,
                    "explainer_identity": self.identity,
                    "cache_key": key,
                    "waterfall_path": str(folder / "waterfall.svg"),
                }
                waterfall(item, folder / "waterfall.svg")
                write_json(folder / "explanation.json", item)
                write_json(
                    folder / "manifest.json",
                    {
                        "status": "complete",
                        "files": {
                            n: file_sha256(folder / n)
                            for n in ("explanation.json", "waterfall.svg")
                        },
                    },
                )
                outputs.append(item)
        return outputs


def explain_report(
    model_path,
    prepared,
    config,
    report_path,
    output_root,
    cache_root,
    scope="sample",
    limit=None,
    run_id=None,
    progress=print,
):
    config.require_ready("explain")
    report_meta, predictions = load_report(report_path)
    if report_meta["model_metadata_sha256"] != file_sha256(
        model_path / "metadata.json"
    ):
        raise ValueError("Explanation report and models differ")
    engine = ExplanationEngine(model_path, prepared, config.shap, cache_root)
    if report_meta["prepared_metadata_sha256"] != file_sha256(
        prepared / "metadata.json"
    ):
        raise ValueError("Explanation report preparation differs")
    split = report_meta["split"]
    preparation = verify_prepared(engine.bundle, prepared)
    if (
        file_sha256(prepared / f"{split}_features.parquet")
        != preparation["files"][f"{split}_features.parquet"]
    ):
        raise ValueError("Explanation feature fingerprint mismatch")
    if split == "test" and engine.bundle.metadata.get("stage") != "validation_selected":
        raise ValueError("Test explanations require frozen selection")
    if scope not in ("sample", "full", "preview"):
        raise ValueError("Unknown explanation coverage")
    count = (
        len(predictions)
        if scope == "full"
        else min(
            config.shap.global_sample_size if scope == "sample" else (limit or 1),
            len(predictions),
        )
    )
    positions = (
        np.arange(len(predictions))
        if scope == "full"
        else np.sort(
            np.random.default_rng(config.shap.seed).choice(
                len(predictions), count, replace=False
            )
        )
    )
    # Fetch matching rows in saved split order without materializing all features.
    selected_ids = predictions.iloc[positions].transaction_id.tolist()
    wanted = set(predictions.iloc[positions].source_row_number.tolist())
    pieces = []
    for batch in pq.ParquetFile(prepared / f"{split}_features.parquet").iter_batches(
        batch_size=25000
    ):
        frame = batch.to_pandas()
        mask = frame.source_row_number.isin(wanted)
        if mask.any():
            pieces.append(frame.loc[mask])
    features = pd.concat(pieces, ignore_index=True)
    if (
        features.source_row_number.tolist()
        != predictions.iloc[positions].source_row_number.tolist()
    ):
        raise ValueError("Explanation record alignment failed")
    output = new_run(output_root, "explain", run_id)
    manifest = {
        "schema_version": 1,
        "status": "running",
        "scope": scope,
        "split": split,
        "rows_requested": count,
        "population_rows": len(predictions),
        "sample_seed": config.shap.seed,
        "model_metadata_sha256": file_sha256(model_path / "metadata.json"),
        "report_metadata_sha256": file_sha256(report_path / "metadata.json"),
        "reference": engine.identity,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output / "metadata.json", manifest)
    totals = {m: np.zeros(len(FEATURE_COLUMNS)) for m in ("rf", "rf_smote")}
    completed = 0
    try:
        with (output / "explanations.jsonl").open("w", encoding="utf-8") as handle:
            for i in range(len(features)):
                entries = engine.explain(
                    features.iloc[[i]].loc[:, FEATURE_COLUMNS],
                    [selected_ids[i]],
                    report_meta["run_id"],
                )
                for entry in entries:
                    handle.write(json.dumps(entry, allow_nan=False) + "\n")
                    totals[entry["model"]] += np.abs(
                        [f["contribution"] for f in entry["features"]]
                    )
                completed += 1
                progress(f"Explained {completed:,}/{count:,} records for both models")
        selected = predictions.iloc[positions][
            ["transaction_id", "source_row_number", "actual_label"]
        ]
        selected.to_csv(output / "explained_records.csv", index=False)
        summary = {
            "scope": scope,
            "rows": count,
            "population_rows": len(predictions),
            "actual_class_counts": {
                str(int(k)): int(v)
                for k, v in selected.actual_label.value_counts().items()
            },
            "mean_absolute_shap": {
                m: dict(zip(FEATURE_COLUMNS, (v / count).tolist()))
                for m, v in totals.items()
            },
            "note": "Sample summary only; not an explanation of every population record"
            if scope != "full"
            else "All records in this evaluation report",
        }
        write_json(output / "global_summary.json", summary)
        manifest.update(
            status="complete",
            rows_completed=completed,
            files={
                p.name: file_sha256(p)
                for p in output.iterdir()
                if p.name != "metadata.json"
            },
        )
        write_json(output / "metadata.json", manifest)
        return output
    except BaseException as exc:
        manifest.update(
            status="failed", rows_completed=completed, failure_type=type(exc).__name__
        )
        write_json(output / "metadata.json", manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("models", "prepared", "report"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--scope", choices=["sample", "full", "preview"], default="sample"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Preview only; no effect on the approved global sample",
    )
    parser.add_argument("--run-id")
    args = parser.parse_args()
    try:
        if args.limit is not None and (args.scope != "preview" or args.limit < 1):
            raise ValueError(
                "--limit must be positive and is available only for preview"
            )
        settings = load_settings()
        resolve = lambda p: p if p.is_absolute() else settings.backend_root / p
        config = load_experiment(resolve(args.config or settings.experiment_config))
        output = explain_report(
            resolve(args.models),
            resolve(args.prepared),
            config,
            resolve(args.report),
            settings.reports_dir,
            settings.artifacts_dir / "explanation_cache",
            args.scope,
            args.limit,
            args.run_id,
            lambda x: print(x, file=sys.stderr, flush=True),
        )
    except (ValueError, OSError, MemoryError) as exc:
        print(json.dumps({"success": False, "error": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps({"success": True, "report": str(output)}))
    return 0
