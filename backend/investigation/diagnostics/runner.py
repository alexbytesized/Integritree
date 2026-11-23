"""Orchestrate an isolated, validation-only diagnostic run."""

import argparse
from datetime import datetime, timedelta, timezone
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import sys
import traceback

import numpy as np
import pandas as pd

from integritree.ml.data import file_sha256, write_json
from integritree.ml.explainability import ExplanationEngine
from integritree.ml.research import new_run

from .analysis import (
    MODELS,
    add_outcomes,
    budget_comparisons,
    findings,
    grid_diagnostics,
    metric_row,
    outcome_counts,
    sample_cases,
    score_distributions,
    subgroup_metrics,
)
from .inputs import input_inventory, load_inputs, read_config, verify_unchanged
from .reporting import save_charts, verify_offline_report, write_report


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def verify_explanation(item, row, model, tolerance):
    if item["model"] != model or item["transaction_id"] != row.transaction_id:
        raise ValueError("SHAP case/model identity mismatch")
    if (
        not np.isfinite(item["output_value"])
        or abs(item["output_value"] - row[f"{model}_risk_score"]) > 1e-12
        or item["predicted_label"] != row[f"{model}_predicted_label"]
        or item["threshold"] != row.threshold
    ):
        raise ValueError("SHAP output differs from saved validation prediction")
    values = [item["base_value"], *[f["contribution"] for f in item["features"]]]
    if (
        not np.isfinite(values).all()
        or abs(sum(values) - item["output_value"]) > tolerance
    ):
        raise ValueError("SHAP contributions fail reconstruction")


def run(root, config_path, run_id=None):
    settings = read_config(config_path)
    (root / "outputs").mkdir(parents=True, exist_ok=True)
    output = new_run(root / "outputs", "investigate", run_id)
    manifest = {
        "schema_version": 1,
        "status": "running",
        "run_id": output.name,
        "created_at": now(),
        "split": "validation",
        "test_used": False,
        "models_retrained": False,
        "thresholds_changed": False,
        "settings": {
            k: str(v) if isinstance(v, Path) else v for k, v in settings.items()
        },
        "configuration_path": str(config_path.resolve()),
        "configuration_sha256": file_sha256(config_path),
    }
    write_json(output / "manifest.json", manifest)

    def progress(message):
        line = f"{now()} {message}"
        with (output / "run.log").open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
        print(line, file=sys.stderr, flush=True)

    try:
        progress(
            "Starting validation-only investigation; input artifacts are read-only"
        )
        manifest["implementation_sha256"] = {
            str(p.relative_to(root)): file_sha256(p)
            for p in [root / "run.py", *sorted((root / "diagnostics").glob("*.py"))]
        }
        manifest["package_versions"] = {
            name: version(name)
            for name in (
                "integritree-backend",
                "numpy",
                "pandas",
                "pyarrow",
                "scikit-learn",
                "imbalanced-learn",
                "shap",
                "matplotlib",
            )
        }
        # Early split check: never follow a test report into model/data loading.
        report_meta = json.loads(
            (settings["report"] / "metadata.json").read_text(encoding="utf-8")
        )
        if (
            report_meta.get("split") != "validation"
            or report_meta.get("official_test") is not False
        ):
            raise ValueError("Investigation accepts validation reports only")
        progress("Snapshotting input fingerprints (excluding test feature files)")
        before = input_inventory(settings)
        manifest["input_sha256"] = before
        write_json(output / "manifest.json", manifest)
        bundle, frame, features, policy, meta = load_inputs(settings, progress)
        if bundle.config.shap.background_size != 200:
            raise ValueError(
                "Investigation requires the approved 200-record SHAP background"
            )
        frame = add_outcomes(frame)
        threshold = meta["threshold"]
        minimum = settings["minimum_denominator"]
        progress(
            "Computing paired outcomes, subgroup support, and threshold diagnostics"
        )
        grid, best, shared = grid_diagnostics(frame)
        sample, coverage = sample_cases(
            frame, settings["sample_per_group"], settings["seed"]
        )
        tables = {
            "outcomes": outcome_counts(frame),
            "overall": pd.DataFrame(
                [
                    metric_row(frame, model, threshold, policy, minimum)
                    for model in MODELS
                ]
            ),
            "subgroups": subgroup_metrics(frame, threshold, policy, minimum),
            "thresholds": grid,
            "best_grid_f1": best,
            "false_alarm_budgets": budget_comparisons(frame),
            "score_distributions": score_distributions(frame),
            "sample_coverage": coverage,
            "sampled_cases": sample,
        }
        for column in ("type", "amount_band"):
            tables[f"outcomes_by_{'amount' if column == 'amount_band' else column}"] = (
                frame.groupby([column, "outcome"], observed=True)
                .size()
                .reset_index(name="rows")
            )
        (output / "tables").mkdir()
        frame.to_parquet(output / "validation_details.parquet", index=False)
        manifest.update(
            rows=len(frame),
            threshold=threshold,
            model_run_id=meta["model_run_id"],
            shared_grid_diagnostic=shared,
            shap_policy=bundle.config.shap.model_dump(),
            shap_scope="sampled outcome/type cases; not population importance",
        )
        write_json(output / "manifest.json", manifest)
        progress(
            f"Initializing shared SHAP engine for {len(sample)} cases / {2 * len(sample)} explanations"
        )
        engine = ExplanationEngine(
            settings["models"],
            settings["prepared"],
            bundle.config.shap,
            root / "runtime" / "shap",
            bundle=bundle,
        )
        explanations, contributions = {}, []
        for i, row in sample.iterrows():
            row_id = int(row.source_row_number)
            items = engine.explain(
                features.loc[[row_id]],
                [row.transaction_id],
                analysis_id="validation_investigation_v1",
            )
            if len(items) != 2 or {item["model"] for item in items} != set(MODELS):
                raise ValueError("SHAP must return one explanation per model per case")
            for item in items:
                model = item["model"]
                verify_explanation(
                    item, row, model, bundle.config.shap.additivity_tolerance
                )
                destination = output / "explanations" / str(row_id) / model
                destination.mkdir(parents=True)
                shutil.copyfile(item["waterfall_path"], destination / "waterfall.svg")
                portable = dict(
                    item, waterfall_path=f"explanations/{row_id}/{model}/waterfall.svg"
                )
                write_json(destination / "explanation.json", portable)
                explanations[(row_id, model)] = portable
                for feature in item["features"]:
                    contributions.append(
                        {
                            "source_row_number": row_id,
                            "transaction_id": row.transaction_id,
                            "outcome": row.outcome,
                            "type": row.type,
                            "actual_label": int(row.actual_label),
                            "model": model,
                            "base_value": item["base_value"],
                            "output_value": item["output_value"],
                            **feature,
                        }
                    )
            progress(f"Explained case {i + 1}/{len(sample)} (both models)")
        tables["shap_contributions"] = pd.DataFrame(contributions)
        for name, table in tables.items():
            table.to_csv(
                output / "tables" / f"{name}.csv", index=False, float_format="%.17g"
            )
        progress("Rendering charts and checking input fingerprints remain unchanged")
        save_charts(output, frame, grid, tables["subgroups"])
        verify_unchanged(before)
        manifest["inputs_unchanged"] = True
        manifest["findings"] = findings(frame, tables["subgroups"], minimum)
        manifest["sampled_transactions"] = len(sample)
        manifest["paired_explanations"] = len(explanations)
        context = {
            **manifest,
            **settings,
            "samples": sample.to_dict("records"),
            "background_size": bundle.config.shap.background_size,
        }
        write_report(output, context, tables, explanations)
        verify_offline_report(output)
        progress(
            "Completed: report, local links, SHAP reconstruction, and input preservation verified"
        )
        manifest.update(
            status="complete", completed_at=now(), offline_links_verified=True
        )
        manifest["files"] = {
            p.relative_to(output).as_posix(): file_sha256(p)
            for p in sorted(output.rglob("*"))
            if p.is_file() and p.name != "manifest.json"
        }
        write_json(output / "manifest.json", manifest)
        return output
    except BaseException as exc:
        # Preserve useful diagnostics, but never label partial output as a completed report.
        manifest.update(
            status="failed", failed_at=now(), error=f"{type(exc).__name__}: {exc}"
        )
        progress(f"FAILED: {manifest['error']}")
        (output / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        write_json(output / "manifest.json", manifest)
        (output / "report.html").write_text(
            '<!doctype html><html lang="en"><meta charset="utf-8"><title>Incomplete investigation</title>'
            "<h1>Investigation incomplete</h1><p>This run failed or was interrupted. "
            "Do not treat its partial results as a completed report.</p>"
            '<p>See <a href="manifest.json">manifest</a> and <a href="run.log">log</a>.</p></html>',
            encoding="utf-8",
        )
        raise


def main(root):
    parser = argparse.ArgumentParser(
        description="Investigate saved RF/RF-SMOTE validation errors; no training."
    )
    parser.add_argument("--config", type=Path, default=root / "config.yaml")
    parser.add_argument("--run-id")
    args = parser.parse_args()
    try:
        output = run(root, args.config.resolve(), args.run_id)
    except (Exception, KeyboardInterrupt) as exc:
        print(
            json.dumps({"success": False, "error": f"{type(exc).__name__}: {exc}"}),
            file=sys.stderr,
            flush=True,
        )
        return 2
    print(
        json.dumps(
            {
                "success": True,
                "report": str(output / "report.html"),
                "manifest": str(output / "manifest.json"),
            }
        ),
        flush=True,
    )
    return 0
