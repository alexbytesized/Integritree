"""Evaluate the six saved forest pairs on an already engineered, labeled CSV."""

import argparse
import csv
import gc
import html
import os
import shutil
import sys
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "src"))

import joblib
import numpy as np
import pandas as pd
from compare_cutoffs import metric_row, source_inventory, verify_grid
from experiment import (
    KINDS,
    bundle_inventory,
    make_cohorts,
    predict,
    read_json,
    sample_positions,
    shap_values,
    time_summary,
    time_sweep,
    verify_unchanged,
)

from integritree.config import EvaluationConfig
from integritree.ml.artifacts import load_bundle
from integritree.ml.data import file_sha256, write_json
from integritree.ml.features import FEATURE_COLUMNS
from integritree.ml.inference import feature_matrix
from integritree.ml.preprocessing import FittedPreprocessor
from integritree.ml.research import new_run
from integritree.ml.threshold_search import percent_grid_thresholds


def load_csv(path):
    """Preserve rows/duplicates and feature values; labels never enter inference."""
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle), [])
    if len(set(header)) != len(header):
        raise ValueError("Duplicate CSV column names")
    if set(header) != set(FEATURE_COLUMNS + ["isFraud"]):
        raise ValueError("CSV must contain the 11 engineered features and isFraud")
    frame = pd.read_csv(path, encoding="utf-8-sig", float_precision="round_trip")
    values = feature_matrix(frame[FEATURE_COLUMNS])
    labels = pd.to_numeric(frame.isFraud, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(labels).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError("isFraud must contain only binary labels 0 and 1")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Both label classes are required for cutoff selection")
    return values, labels.astype("uint8"), np.arange(1, len(frame) + 1)


def load_pair(item, source):
    if item["source"]:
        bundle = load_bundle(Path(item["source"]))
        return bundle.models
    return {
        kind: joblib.load(source / item["candidate"] / f"{kind}.joblib")
        for kind in KINDS
    }


def check_counts(row, labels, scores):
    flagged = scores >= row["threshold"]
    expected = {
        "tp": int(np.sum(flagged & (labels == 1))),
        "fp": int(np.sum(flagged & (labels == 0))),
        "tn": int(np.sum(~flagged & (labels == 0))),
        "fn": int(np.sum(~flagged & (labels == 1))),
    }
    if any(row[k] != v for k, v in expected.items()) or sum(expected.values()) != len(
        labels
    ):
        raise ValueError("Confusion matrix verification failed")


def comparison_tables(output, rows, summaries, source, prior_cutoffs):
    current = pd.DataFrame(rows)
    shap = pd.DataFrame(summaries)
    uniform = shap.query("cohort == 'uniform_validation'")
    combined = current.merge(
        uniform[["candidate", "model", "mean_time_abs_pp", "time_share_abs"]],
        on=["candidate", "model"],
        validate="many_to_one",
    )
    previous = pd.concat(
        [
            pd.read_csv(source / "metrics.csv", float_precision="round_trip").assign(
                mode="fixed_43"
            ),
            pd.read_csv(
                prior_cutoffs / "metrics_selected.csv", float_precision="round_trip"
            ).assign(mode="selected"),
        ],
        ignore_index=True,
    )
    columns = ["threshold", "precision", "recall", "f1", "pr_auc", "tp", "fp"]
    changes = current.merge(
        previous[["candidate", "model", "mode"] + columns],
        on=["candidate", "model", "mode"],
        suffixes=("", "_previous"),
        validate="one_to_one",
    )
    if len(changes) != len(current):
        raise ValueError("Incomplete comparison with previous results")
    for name in columns:
        changes[f"delta_{name}"] = changes[name] - changes[f"{name}_previous"]
    old_shap = pd.read_csv(source / "time_shap.csv", float_precision="round_trip")
    shap_changes = shap.merge(
        old_shap,
        on=["candidate", "model", "cohort"],
        suffixes=("", "_previous"),
        validate="one_to_one",
    )
    for name in ("mean_time_abs_pp", "time_share_abs"):
        shap_changes[f"delta_{name}"] = (
            shap_changes[name] - shap_changes[f"{name}_previous"]
        )
    for name, table in {
        "comparison": combined,
        "changes_vs_previous": changes,
        "shap_changes_vs_previous": shap_changes,
    }.items():
        table.to_csv(output / f"{name}.csv", index=False, float_format="%.17g")
    return combined, changes


def display_table(frame):
    fields = [
        "candidate",
        "model",
        "threshold",
        "precision",
        "recall",
        "f1",
        "pr_auc",
        "tp",
        "fp",
        "mean_time_abs_pp",
        "time_share_abs",
    ]
    table = frame[fields].copy()
    table["candidate"] = table.candidate.str.replace(
        "depth_", "", regex=False
    ).str.replace("_leaf_", " / ", regex=False)
    table["model"] = table.model.map({"rf": "RF", "rf_smote": "RF-SMOTE"})
    for name in ("threshold", "precision", "recall", "time_share_abs"):
        table[name] = table[name].map(
            lambda v: "undefined" if pd.isna(v) else f"{v:.2%}"
        )
    for name in ("f1", "pr_auc", "mean_time_abs_pp"):
        table[name] = table[name].map(
            lambda v: "undefined" if pd.isna(v) else f"{v:.4f}"
        )
    table.columns = [
        "Depth / leaf",
        "Model",
        "Cutoff",
        "Precision",
        "Recall",
        "F1",
        "PR-AUC (AP)",
        "Fraud detected",
        "False alarms",
        "Time contribution (pp)",
        "Time share",
    ]
    return table


def render(output, combined, changes, manifest):
    note = (
        f"Experimental validation using {Path(manifest['validation_csv']).name}: "
        f"{manifest['rows']:,} rows, {manifest['actual_fraud']:,} fraud labels. "
        "All 12 saved models were reused; no retraining or deployment. "
        "Features are used as supplied without rescaling or deduplication. "
        "Selected cutoffs maximize the pair's average F1 on the 1%–100% grid; "
        "ties prefer closest to 50%, then higher. These are validation results, not an independent test."
    )
    explanation = (
        "Time contribution (pp) is 100 × mean(|SHAP hour| + |SHAP day|). "
        "Time share is sum of absolute hour/day SHAP divided by sum of all absolute feature SHAP. "
        "Both displayed time measures use the same 200 uniformly sampled CSV rows across all models "
        "and the original 200-row training background. They are unchanged by cutoff selection. "
        "Other cohort summaries are in time_shap.csv. Cohorts overlap; flagged cohorts are frozen "
        "using the current baseline at 43%, not reselected at each optimum. "
        "Row order differs between datasets, so seeded samples need not identify the same records. "
        "SHAP changes compare sampled summaries, not matched-record effects. "
        "The hour/day sweep is a sensitivity diagnostic, not labeled accuracy evidence. "
        "Dataset provenance and overlap were not investigated for this run."
    )
    lines = ["# CSV validation experiment", "", note, "", explanation, ""]
    body = [
        "<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>",
        "<title>CSV validation experiment</title><style>body{font:16px system-ui;margin:2rem;color:#172838}table{border-collapse:collapse;font-size:14px}td,th{padding:8px;border:1px solid #ccc}section{overflow:auto}p{max-width:100ch}</style>",
        "<h1>CSV validation experiment</h1><p>"
        + html.escape(note)
        + "</p><p>"
        + html.escape(explanation)
        + "</p>",
    ]
    for mode, title in (
        ("selected", "Each pair's selected cutoff"),
        ("fixed_43", "Fixed 43% cutoff"),
    ):
        table = display_table(combined[combined["mode"] == mode])
        lines += [
            f"## {title}",
            "",
            "| " + " | ".join(table.columns) + " |",
            "|" + "---|" * len(table.columns),
        ]
        lines += [
            "| " + " | ".join(map(str, row)) + " |"
            for row in table.itertuples(index=False, name=None)
        ]
        lines.append("")
        body += [
            "<h2>" + html.escape(title) + "</h2><section>",
            table.to_html(index=False, escape=True),
            "</section>",
        ]
    delta = changes.query("mode == 'selected'")[
        [
            "candidate",
            "model",
            "threshold_previous",
            "threshold",
            "f1_previous",
            "f1",
            "delta_f1",
            "delta_tp",
            "delta_fp",
        ]
    ]
    body += [
        "<h2>Compared with previous selected-cutoff results</h2><p>Both evaluations select their own cutoff. Use changes_vs_previous.csv fixed_43 rows to compare at an unchanged cutoff.</p><section>",
        delta.to_html(index=False, float_format=lambda v: f"{v:.6f}"),
        "</section></html>",
    ]
    lines += [
        "Previous results and their changes are in changes_vs_previous.csv and report.html. PR-AUC uses average precision.",
        "",
        "See manifest.json and verification.json for completion status and reproducibility details.",
    ]
    (output / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "report.html").write_text("\n".join(body), encoding="utf-8")


def run(csv_path, source, prior_cutoffs, run_id=None, output_root=None):
    started = time.monotonic()
    csv_path, source, prior_cutoffs = (
        Path(p).resolve() for p in (csv_path, source, prior_cutoffs)
    )
    output = new_run(output_root or ROOT / "outputs", "csv_validation", run_id)
    os.environ["NUMBA_CACHE_DIR"] = str(ROOT / "runtime" / "numba")
    os.environ["MPLCONFIGDIR"] = str(ROOT / "runtime" / "matplotlib")
    manifest = {
        "schema_version": 1,
        "kind": "isolated_csv_validation",
        "status": "running",
        "created_at": datetime.now(UTC).isoformat(),
        "validation_csv": str(csv_path),
        "source": str(source),
        "prior_cutoffs": str(prior_cutoffs),
        "new_fits": 0,
        "official_test_evaluated": False,
        "data_provenance_audited": False,
        "completed": [],
        "identity": "source_row_number is the 1-based CSV data-row position (header excluded); namespace is validation_csv_sha256",
        "selection": {
            "grid_percent": list(range(1, 101)),
            "objective": "mean_validation_f1",
            "tie_policy": "closest_to_50_then_higher",
            "score_ties": "fraud",
        },
        "packages": {
            p: version(p)
            for p in ("numpy", "pandas", "scikit-learn", "shap", "pyarrow", "joblib")
        },
    }

    def save():
        write_json(output / "manifest.json", manifest)

    def log(message):
        print(message, flush=True)
        with (output / "run.log").open("a", encoding="utf-8") as handle:
            handle.write(datetime.now(UTC).isoformat() + " " + message + "\n")

    save()
    try:
        log("Loading saved experiment references and supplied CSV; no fitting")
        original, inventory = source_inventory(source)
        plan = read_json(source / "plan.json")["candidates"]
        if plan != original["plan"] or len(plan) != 6:
            raise ValueError("Expected the original six-pair experiment")
        settings = original["settings"]
        policy = EvaluationConfig.model_validate(
            read_json(source / "plan.json")["fixed_configuration"]["evaluation"]
        )
        prior_manifest = read_json(prior_cutoffs / "manifest.json")
        if (
            prior_manifest["status"] != "complete"
            or Path(prior_manifest["source"]).resolve() != source
        ):
            raise ValueError("Previous cutoff results must belong to the original run")
        for path in (
            csv_path,
            prior_cutoffs / "manifest.json",
            prior_cutoffs / "metrics_selected.csv",
        ):
            inventory[str(path)] = file_sha256(path)
        if (
            inventory[str(prior_cutoffs / "metrics_selected.csv")]
            != prior_manifest["files"]["metrics_selected.csv"]
        ):
            raise ValueError("Previous metric fingerprint mismatch")
        for item in plan:
            if item["source"]:
                inventory.update(bundle_inventory(Path(item["source"])))
        implementation = {
            str(p): file_sha256(p)
            for p in [
                *ROOT.glob("*.py"),
                *(ROOT.parent / "src" / "integritree" / "ml").glob("*.py"),
            ]
        }
        X, y, ids = load_csv(csv_path)
        background = pd.read_parquet(source / "shap_background.parquet")[
            FEATURE_COLUMNS
        ].to_numpy(dtype=float)
        preprocessor = FittedPreprocessor.load(source / "preprocessing.json")
        if len(background) != 200:
            raise ValueError("Expected the original 200-row SHAP background")
        for name in ("shap_background.parquet", "preprocessing.json"):
            shutil.copyfile(source / name, output / name)
        manifest.update(
            rows=len(y),
            actual_fraud=int(y.sum()),
            legitimate=int((y == 0).sum()),
            validation_csv_sha256=inventory[str(csv_path)],
            input_sha256=inventory,
            implementation_sha256=implementation,
            plan=plan,
            sampling=settings,
        )
        save()
        rows, summaries, sweeps, checks, decisions = [], [], [], [], []
        cohorts = chosen = None
        sensitivity = sample_positions(
            np.arange(len(y)), settings["sensitivity_rows"], settings["seed"]
        )
        pd.DataFrame({"source_row_number": ids[sensitivity]}).to_csv(
            output / "sensitivity_cohort.csv", index=False
        )
        for item in plan:
            name = item["candidate"]
            folder = output / name
            folder.mkdir()
            log(f"Scoring {name} on {len(y):,} CSV rows")
            models = load_pair(item, source)
            for model in models.values():
                if (
                    model.max_depth,
                    model.min_samples_leaf,
                    model.n_estimators,
                    model.random_state,
                ) != (item["max_depth"], item["min_samples_leaf"], 100, 42):
                    raise ValueError(
                        "Saved model parameters do not match the configuration"
                    )
            scores = {kind: predict(models[kind], X) for kind in KINDS}
            if cohorts is None:
                if name != "depth_10_leaf_1":
                    raise ValueError("Current baseline must be evaluated first")
                cohorts = make_cohorts(y, scores, settings)
                chosen = np.unique(np.concatenate(list(cohorts.values())))
                pd.DataFrame(
                    [
                        {
                            "cohort": cohort,
                            "source_row_number": int(ids[i]),
                            "actual_label": int(y[i]),
                        }
                        for cohort, positions in cohorts.items()
                        for i in positions
                    ]
                ).to_csv(output / "shap_cohorts.csv", index=False)
                manifest["shap_unique_rows"] = len(chosen)
            decision, grid = percent_grid_thresholds(
                y, scores["rf"], scores["rf_smote"]
            )
            grid_check = verify_grid(y, scores, decision, grid)
            write_json(
                folder / "decision.json", decision | {"verification": grid_check}
            )
            grid.to_csv(folder / "thresholds.csv", index=False, float_format="%.17g")
            decisions.append({"candidate": name} | decision)
            predictions = pd.DataFrame({"source_row_number": ids, "actual_label": y})
            for kind in KINDS:
                model = models[kind]
                predictions[f"{kind}_score"] = scores[kind]
                for mode, cutoff in (
                    ("fixed_43", 0.43),
                    ("selected", decision["threshold"]),
                ):
                    row = metric_row(y, scores[kind], cutoff, policy, name, kind, mode)
                    check_counts(row, y, scores[kind])
                    rows.append(row)
                    predictions[f"{kind}_{mode}_label"] = (
                        scores[kind] >= cutoff
                    ).astype("uint8")
                log(
                    f"Explaining {name}/{kind} on {len(chosen)} fixed CSV rows; cutoff {decision['threshold']:.0%}"
                )
                phi, base, error = shap_values(model, background, X[chosen])
                explanation = pd.DataFrame(phi, columns=FEATURE_COLUMNS)
                explanation.insert(0, "source_row_number", ids[chosen])
                explanation["base_score"], explanation["score"] = (
                    base,
                    scores[kind][chosen],
                )
                explanation.to_parquet(folder / f"{kind}_shap.parquet", index=False)
                for cohort, positions in cohorts.items():
                    summaries.append(
                        {"candidate": name, "model": kind, "cohort": cohort}
                        | time_summary(phi[np.searchsorted(chosen, positions)])
                    )
                sweep, per_row = time_sweep(
                    model, X[sensitivity], ids[sensitivity], preprocessor
                )
                per_row["crosses_selected_cutoff"] = (
                    per_row.minimum_score < decision["threshold"]
                ) & (per_row.maximum_score >= decision["threshold"])
                sweep.to_parquet(folder / f"{kind}_time_sweep.parquet", index=False)
                per_row.to_csv(
                    folder / f"{kind}_time_sensitivity.csv",
                    index=False,
                    float_format="%.17g",
                )
                sweeps.append(
                    {
                        "candidate": name,
                        "model": kind,
                        "rows": len(per_row),
                        "mean_range_pp": float(per_row.range_pp.mean()),
                        "median_range_pp": float(per_row.range_pp.median()),
                        "crosses_at_43": float(per_row.crosses_43_percent.mean()),
                        "crosses_at_selected": float(
                            per_row.crosses_selected_cutoff.mean()
                        ),
                    }
                )
                checks.append(
                    {
                        "candidate": name,
                        "model": kind,
                        "shap_additivity_max_error": error,
                        "confusion_counts_verified": True,
                        "predictions": len(y),
                        "shap_rows": len(chosen),
                        "time_sweep_scores": len(sweep),
                    }
                )
            predictions["selected_threshold"] = decision["threshold"]
            predictions.to_parquet(folder / "predictions.parquet", index=False)
            manifest["completed"].append(name)
            for label, records in (
                ("metrics", rows),
                ("time_shap", summaries),
                ("time_sensitivity", sweeps),
                ("pair_decisions", decisions),
            ):
                pd.DataFrame(records).to_csv(
                    output / f"{label}.csv", index=False, float_format="%.17g"
                )
            save()
            del models, model, scores, predictions
            gc.collect()
        combined, changes = comparison_tables(
            output, rows, summaries, source, prior_cutoffs
        )
        render(output, combined, changes, manifest)
        log(
            "Verifying saved scores, output row counts, SHAP reconstruction, and preservation"
        )
        for item in plan:
            folder = output / item["candidate"]
            frame = pd.read_parquet(folder / "predictions.parquet")
            np.testing.assert_array_equal(frame.source_row_number, ids)
            np.testing.assert_array_equal(frame.actual_label, y)
            for kind in KINDS:
                for mode in ("fixed_43", "selected"):
                    row = next(
                        r
                        for r in rows
                        if r["candidate"] == item["candidate"]
                        and r["model"] == kind
                        and r["mode"] == mode
                    )
                    check_counts(row, y, frame[f"{kind}_score"].to_numpy())
                    np.testing.assert_array_equal(
                        frame[f"{kind}_{mode}_label"],
                        frame[f"{kind}_score"] >= row["threshold"],
                    )
                explanation = pd.read_parquet(folder / f"{kind}_shap.parquet")
                np.testing.assert_array_equal(
                    explanation.source_row_number, ids[chosen]
                )
                np.testing.assert_allclose(
                    explanation.base_score + explanation[FEATURE_COLUMNS].sum(axis=1),
                    frame[f"{kind}_score"].to_numpy()[chosen],
                    rtol=0,
                    atol=1e-6,
                )
        verify_unchanged(inventory)
        verify_unchanged(implementation)
        write_json(
            output / "verification.json",
            {
                "status": "passed",
                "models": checks,
                "thresholds_verified": len(plan) * 100,
                "new_fits": 0,
                "source_preservation_verified": True,
            },
        )
        log("Completed all six pairs; no models or application settings changed")
        manifest.update(
            status="complete",
            completed_at=datetime.now(UTC).isoformat(),
            elapsed_seconds=time.monotonic() - started,
            source_preservation_verified=True,
            files={
                p.relative_to(output).as_posix(): file_sha256(p)
                for p in output.rglob("*")
                if p.is_file() and p.name != "manifest.json"
            },
        )
        save()
        return output
    except BaseException as exc:
        manifest.update(
            status="failed", failure_type=type(exc).__name__, error=str(exc)
        )
        save()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "outputs" / "forest_regularization_20261117",
    )
    parser.add_argument(
        "--prior-cutoffs",
        type=Path,
        default=ROOT / "outputs" / "paired_cutoffs_20261117",
    )
    parser.add_argument("--run-id")
    args = parser.parse_args()
    print(run(args.csv, args.source, args.prior_cutoffs, args.run_id))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
