"""Shared probability-space TreeSHAP with reproducible background, provenance, and coverage."""

import argparse
import hashlib
import json
import os
import sys
import threading
import time
import uuid
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from integritree.config import load_experiment
from integritree.ml.artifacts import load_bundle
from integritree.ml.data import file_sha256, write_json
from integritree.ml.features import FEATURE_COLUMNS
from integritree.ml.inference import feature_matrix, fraud_scores
from integritree.ml.research import load_report, new_run, verify_prepared
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


CHART_VERSION = "contribution_bars_v3"
# Shared with React; see docs/API.md for the pixel coordinate contract.
CHART_GEOMETRY = json.loads(Path(__file__).with_name("shap_geometry.json").read_text())
FEATURE_LABELS = dict(
    zip(
        FEATURE_COLUMNS,
        [
            "Hour of the Day",
            "Day of the Week",
            "Transaction is Cash In",
            "Transaction is Cash Out",
            "Transaction is Debit",
            "Transaction is Payment",
            "Transaction is Transfer",
            "Transaction Log Amount",
            "Transaction Amount is 0",
            "Origin is Merchant",
            "Destination is Merchant",
        ],
    )
)
CHART_TITLE = "Contribution to Fraud Risk Score (Percentage Points)"
INCREASE_COLOR = "#A00000"
DECREASE_COLOR = "#009900"
_WATERFALL_LOCK = threading.Lock()


def chart_description(explanation):
    return (
        "SHAP contribution bar chart in fixed feature order. Green, negative contributions "
        "decrease the fraud risk score; red, positive contributions increase it. "
        "The axis spans -100 to +100 percentage points with gridlines every 10. Blank rows have exactly zero contribution. "
        + explanation.get("narrative", "")
    )


def cached_contribution_chart(
    explanation, directory, *, display_label=None, layout="standalone"
):
    """Render stored values without rerunning SHAP; version every presentation."""
    if layout not in ("modal", "standalone"):
        raise ValueError("Unknown SHAP chart layout")
    identity = {
        key: explanation[key]
        for key in ("model", "transaction_id", "features", "base_value", "output_value")
    }
    identity.update(
        display_label=display_label,
        layout=layout,
        top_positive_contributor=explanation.get("top_positive_contributor"),
    )
    target = Path(directory) / f"{CHART_VERSION}_{digest(identity)}.svg"
    if not target.exists():
        temporary = target.with_name(f"{uuid.uuid4().hex}.svg")
        try:
            waterfall(
                explanation, temporary, display_label=display_label, layout=layout
            )
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    return target


def waterfall(explanation, output, *, display_label=None, layout="standalone"):
    """Legacy entry point for signed contribution charts."""
    if layout not in ("modal", "standalone"):
        raise ValueError("Unknown SHAP chart layout")
    with _WATERFALL_LOCK:
        _render_waterfall(
            explanation, output, display_label=display_label, layout=layout
        )


def _render_waterfall(explanation, output, *, display_label=None, layout="standalone"):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    ranks = {name: i for i, name in enumerate(FEATURE_COLUMNS)}
    features = sorted(
        explanation["features"], key=lambda f: ranks.get(f["feature"], len(ranks))
    )
    values = [f["contribution"] * 100 for f in features]
    names = [
        FEATURE_LABELS.get(f["feature"], f["feature"].replace("_", " "))
        for f in features
    ]
    geometry = CHART_GEOMETRY
    limit = geometry["axisLimit"]
    plot_width = 2 * limit / geometry["tickStep"] * geometry["tickSpacing"]
    plot_height = len(features) * geometry["rowHeight"]
    standalone = layout == "standalone"
    # Modal SVG is only the plot plus endpoint padding and ticks. HTML owns labels.
    left = 420 if standalone else geometry["endpointPadding"]
    top = 110 if standalone else geometry["topPadding"]
    bottom = 250 if standalone else geometry["tickHeight"]
    width = left + plot_width + geometry["endpointPadding"]
    height = top + plot_height + bottom
    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
    ax = fig.add_axes(
        (left / width, bottom / height, plot_width / width, plot_height / height)
    )
    try:
        ax.set_xlim(-limit, limit)
        ax.set_ylim(len(features) - 0.5, -0.5)
        if standalone:
            ax.set_yticks(range(len(features)), names, fontsize=12, fontweight="normal")
        else:
            ax.set_yticks([])
        # Reserve endpoint annotation space between standalone names and the axis.
        ax.tick_params(axis="y", length=0, pad=90, colors="#222222")
        ticks = range(-limit, limit + 1, geometry["tickStep"])
        ax.set_xticks(list(ticks), [str(tick) for tick in ticks], fontweight="normal")
        ax.tick_params(axis="x", length=0, pad=10, labelsize=11, colors="#666666")
        for tick in ticks:
            ax.axvline(
                tick,
                color="#888888" if tick == 0 else "#dddddd",
                linewidth=1.2 if tick == 0 else 0.6,
                zorder=1,
            )
        for spine in ax.spines.values():
            spine.set_visible(False)
        # Correct corner radii for data coordinates, without changing bar endpoints.
        aspect = len(features) / ax.bbox.height * ax.bbox.width / (2 * limit)
        for row, (feature, value) in enumerate(zip(features, values)):
            if value == 0:
                continue
            bar = FancyBboxPatch(
                (min(0, value), row - 0.3),
                abs(value),
                0.6,
                boxstyle=f"round,pad=0,rounding_size={min(abs(value) / 2, 0.6)}",
                mutation_aspect=aspect,
                facecolor=INCREASE_COLOR if value > 0 else DECREASE_COLOR,
                edgecolor="none",
                zorder=3,
            )
            bar.set_gid(f"shap-{feature['feature']}")
            ax.add_patch(bar)
            if abs(value) * plot_width / (2 * limit) < 1:
                # A hairline keeps nonzero effects visible without inflating bar lengths.
                ax.plot(
                    [value, value],
                    [row - 0.3, row + 0.3],
                    color=bar.get_facecolor(),
                    linewidth=0.7,
                    zorder=3,
                )
            ax.annotate(
                f"{value:+.3g} pp",
                (value, row),
                xytext=(-7 if value < 0 else 7, 0),
                textcoords="offset points",
                ha="right" if value < 0 else "left",
                va="center",
                fontsize=11,
                color="#222222",
            )
        if standalone:
            ax.set_title(CHART_TITLE, fontsize=17, fontweight="bold", pad=30)
            ax.text(
                0,
                -55 / plot_height,
                "\u2190 Decrease Fraud Risk Score",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=14,
                color=DECREASE_COLOR,
            )
            ax.text(
                1,
                -55 / plot_height,
                "Increase Fraud Risk Score \u2192",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=14,
                color=INCREASE_COLOR,
            )
            model_label = {"rf": "Benchmark RF", "rf_smote": "RF-SMOTE"}[
                explanation["model"]
            ]
            record_label = (
                explanation["transaction_id"]
                if display_label is None
                else display_label
            )
            if len(record_label) > 55:
                record_label = record_label[:20] + "..." + record_label[-24:]
            fig.suptitle(
                f"{model_label} | {record_label}",
                fontsize=19,
                y=0.97,
                fontweight="bold",
            )
            # The summary has its own axes, aligned with the chart's plotting area.
            summary = fig.add_axes(
                (left / width, 22 / height, plot_width / width, 140 / height)
            )
            summary.set_axis_off()

            def card(x, y, width, label, value):
                summary.add_patch(
                    FancyBboxPatch(
                        (x, y),
                        width,
                        0.4,
                        boxstyle="round,pad=0,rounding_size=0.016",
                        mutation_aspect=summary.bbox.width / summary.bbox.height,
                        linewidth=0.8,
                        edgecolor="#dddddd",
                        facecolor="white",
                    )
                )
                summary.text(
                    x + 0.018,
                    y + 0.20,
                    label,
                    fontsize=14,
                    va="center",
                    weight="normal",
                    color="#292929",
                )
                summary.text(
                    x + width - 0.018,
                    y + 0.20,
                    value,
                    fontsize=14,
                    va="center",
                    ha="right",
                    color="#444444",
                )

            card(0, 0.57, 0.46, "Reference", f"{explanation['base_value'] * 100:.2f}%")
            card(
                0.54, 0.57, 0.46, "Output", f"{explanation['output_value'] * 100:.2f}%"
            )
            top = explanation.get("top_positive_contributor", {})
            name = top.get("feature")
            top_label = (
                "No transaction details meaningfully increased the score."
                if top.get("status") == "no_positive_contributor"
                else FEATURE_LABELS.get(name, name or "Unavailable")
            )
            card(0, 0.02, 1, "Top risk-increasing contributor:", top_label)
        fig.savefig(output, format=Path(output).suffix.lstrip("."))
    finally:
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
                    "chart_description": chart_description({"narrative": narrative}),
                    "chart_version": CHART_VERSION,
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
        "created_at": datetime.now(UTC).isoformat(),
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
