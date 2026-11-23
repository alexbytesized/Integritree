"""Three sequential isolated fits, matched evaluation, and verified evidence."""

import argparse
from datetime import datetime, timezone
import gc
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import sys
import traceback

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from integritree.ml.artifacts import ModelBundle, PACKAGES
from integritree.ml.data import file_sha256, write_json
from integritree.ml.explainability import ExplanationEngine
from integritree.ml.features import FEATURE_COLUMNS
from integritree.ml.inference import feature_matrix, fraud_scores
from integritree.ml.preparation import load_prepared_split
from integritree.ml.research import new_run
from integritree.ml.selection_utils import release_candidate_memory
from integritree.ml.training import memory_preflight, resample_training, validate_labels
from investigation.diagnostics.analysis import (
    outcome_counts,
    sample_cases,
    score_distributions,
    subgroup_metrics,
)
from investigation.diagnostics.inputs import (
    input_inventory,
    load_inputs,
    verify_unchanged,
)
from investigation.diagnostics.runner import verify_explanation

from .core import (
    choose_cutoff,
    load_model,
    make_pair,
    overview,
    predict,
    read_config,
    save_model_metadata,
    verify_reference,
    verify_training_ids,
)
from .reporting import page, save_pair_report, save_summary


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def train_one(
    folder, values, labels, config, reference, parameters, prepared, progress
):
    folder.mkdir()
    write_json(
        folder / "metadata.json", {"status": "running", "kind": "isolated_smote_ratio"}
    )
    progress(
        f"Resampling original training rows only at ratio {config.smote.sampling_ratio:.8g}"
    )
    resampled, targets, audit = resample_training(values, labels, config)
    audit.update(
        original_rows_preserved=True,
        original_labels_preserved=True,
        validation_resampled=False,
        neighbors=config.smote.k_neighbors,
        sampling_seed=config.seeds.smote,
    )
    pd.DataFrame(resampled[len(values) :], columns=FEATURE_COLUMNS).to_parquet(
        folder / "synthetic_features.parquet", index=False
    )
    write_json(folder / "training_audit.json", audit)
    progress(
        f"Fitting one forest: {len(resampled):,} rows, {parameters['n_estimators']} trees, one worker"
    )
    model = RandomForestClassifier(**parameters).fit(resampled, targets)
    del resampled, targets
    gc.collect()
    expected = fraud_scores(model, values[:128])
    joblib.dump(model, folder / "rf_smote.joblib", compress=3)
    reloaded = joblib.load(folder / "rf_smote.joblib")
    actual = fraud_scores(reloaded, values[:128])
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
    verification = {
        "passed": True,
        "rows": len(expected),
        "atol": 1e-12,
        "max_absolute_error": float(np.max(np.abs(actual - expected))),
    }
    shutil.copyfile(prepared / "preprocessing.json", folder / "preprocessing.json")
    metadata = save_model_metadata(
        folder, reference, parameters, config, audit, verification
    )
    checked = load_model(folder, reference.metadata["prepared_metadata_sha256"])
    progress(
        f"Saved isolated model; {audit['synthetic_rows']:,} synthetic records; reload verified"
    )
    return checked, metadata, audit, model


def analyze_pair(
    output,
    denominator,
    reference,
    scores,
    features,
    bundle,
    model,
    model_metadata,
    settings,
    policy,
    root,
    progress,
):
    selected, grid = choose_cutoff(reference, scores)
    cutoff = selected["threshold"]
    frames = {
        "43": make_pair(reference, scores, 0.43),
        "selected": make_pair(reference, scores, cutoff),
    }
    progress(f"Ratio 1:{denominator}: selected common cutoff {cutoff:.0%}")
    minimum = settings["minimum_denominator"]
    tables = {"thresholds": grid}
    (output / "tables").mkdir()
    for mode, frame in frames.items():
        frame.to_parquet(output / f"predictions_{mode}.parquet", index=False)
        # CSV contains exact identities, truth, scores and decisions without repeating attributes.
        cols = [
            "source_row_number",
            "transaction_id",
            "actual_label",
            "threshold",
            "outcome",
        ]
        cols += [
            f"{m}_{suffix}"
            for m in ("rf", "rf_smote", "reference_100")
            for suffix in ("risk_score", "predicted_label")
        ]
        frame[cols].to_csv(
            output / f"predictions_{mode}.csv", index=False, float_format="%.17g"
        )
        tables[f"overall_{mode}"] = overview(frame, policy, minimum, denominator, mode)
        tables[f"outcomes_{mode}"] = outcome_counts(frame)
        tables[f"subgroups_{mode}"] = subgroup_metrics(
            frame, float(frame.threshold.iloc[0]), policy, minimum
        )
        for column in ("type", "amount_band", "hour", "day"):
            tables[f"outcomes_by_{column}_{mode}"] = (
                frame.groupby([column, "outcome"], observed=True)
                .size()
                .reset_index(name="rows")
            )
    frame = frames["selected"]
    sample, coverage = sample_cases(
        frame, settings["sample_per_group"], settings["seed"]
    )
    tables.update(
        sampled_cases=sample,
        sample_coverage=coverage,
        score_distributions=score_distributions(frame),
    )
    config = bundle.config.model_copy(deep=True)
    config.smote.sampling_ratio = 1 / denominator
    config.scoring.threshold = cutoff
    paired = ModelBundle(
        models={"rf": bundle.models["rf"], "rf_smote": model},
        preprocessor=bundle.preprocessor,
        config=config,
        metadata=model_metadata,
    )
    progress(f"Explaining {len(sample)} sampled validation cases with both models")
    engine = ExplanationEngine(
        output / "model",
        settings["prepared"],
        config.shap,
        root / "runtime" / "shap",
        bundle=paired,
    )
    explanations, contributions = {}, []
    for index, row in sample.iterrows():
        row_id = int(row.source_row_number)
        items = engine.explain(
            features.loc[[row_id]],
            [row.transaction_id],
            analysis_id=f"ratio_{denominator}_cutoff_{cutoff:.2f}",
        )
        if len(items) != 2 or {x["model"] for x in items} != {"rf", "rf_smote"}:
            raise ValueError("Expected paired SHAP explanations")
        for item in items:
            name = item["model"]
            verify_explanation(item, row, name, config.shap.additivity_tolerance)
            target = output / "explanations" / str(row_id) / name
            target.mkdir(parents=True)
            shutil.copyfile(item["waterfall_path"], target / "waterfall.svg")
            portable = dict(
                item,
                waterfall_path=f"explanations/{row_id}/{name}/waterfall.svg",
                ratio_denominator=denominator,
            )
            write_json(target / "explanation.json", portable)
            explanations[(row_id, name)] = portable
            for feature in item["features"]:
                contributions.append(
                    {
                        "source_row_number": row_id,
                        "model": name,
                        "base_value": item["base_value"],
                        "output_value": item["output_value"],
                        **feature,
                    }
                )
        if (index + 1) % 10 == 0 or index + 1 == len(sample):
            progress(
                f"Ratio 1:{denominator}: explained {index + 1}/{len(sample)} cases"
            )
    tables["shap_contributions"] = pd.DataFrame(contributions)
    for name, table in tables.items():
        table.to_csv(
            output / "tables" / f"{name}.csv", index=False, float_format="%.17g"
        )
    save_pair_report(output, denominator, frame, frames["43"], tables, explanations)
    write_json(
        output / "comparison.json",
        {
            "status": "complete",
            "ratio_denominator": denominator,
            "selected": selected,
            "rows": len(reference),
            "sampled_cases": len(sample),
            "explanations": len(explanations),
            "same_cutoff_for_all_models": True,
            "same_validation_identities": True,
            "offline_links_verified": True,
        },
    )
    return tables["overall_43"], tables["overall_selected"], selected


def run(root, config_path, run_id=None):
    settings = read_config(config_path)
    output = new_run(root / "outputs", "smote_ratios", run_id)
    manifest = {
        "schema_version": 1,
        "kind": "isolated_smote_ratio_experiment",
        "status": "running",
        "created_at": now(),
        "run_id": output.name,
        "test_used": False,
        "deployment_changed": False,
        "new_fits_started": 0,
        "settings": {
            k: str(v) if isinstance(v, Path) else v for k, v in settings.items()
        },
        "configuration_sha256": file_sha256(config_path),
        "completed_ratios": [],
    }
    write_json(output / "manifest.json", manifest)
    (output / "report.html").write_text(
        page(
            "Incomplete experiment",
            "<h1>Experiment incomplete</h1><p>Run is in progress. Only a complete manifest denotes a finished comparison.</p>",
        ),
        encoding="utf-8",
    )

    def progress(message):
        line = f"{now()} {message}"
        with (output / "run.log").open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
        print(line, file=sys.stderr, flush=True)

    try:
        progress(
            "Starting three approved ordinary-SMOTE fits; validation only; references unchanged"
        )
        report = json.loads(
            (settings["report"] / "metadata.json").read_text(encoding="utf-8")
        )
        if (
            report.get("split") != "validation"
            or report.get("official_test") is not False
        ):
            raise ValueError("Only validation reports are accepted")
        code_files = [
            root / "run.py",
            *sorted((root / "ratio_experiment").glob("*.py")),
            *sorted((root.parent / "investigation" / "diagnostics").glob("*.py")),
            *sorted((root.parent / "src" / "integritree" / "ml").glob("*.py")),
        ]
        code_before = {str(p.resolve()): file_sha256(p) for p in code_files}
        before = input_inventory(settings)
        manifest.update(
            input_sha256=before,
            implementation_sha256=code_before,
            package_versions={
                p: version(p) for p in (*PACKAGES, "shap", "matplotlib", "PyYAML")
            },
        )
        write_json(output / "manifest.json", manifest)
        shutil.copyfile(config_path, output / "configuration.yaml")
        bundle, reference, features, policy, report_meta = load_inputs(
            settings, progress
        )
        parameters = verify_reference(bundle)
        progress("Replaying both saved reference scores on every validation record")
        for name in ("rf", "rf_smote"):
            np.testing.assert_allclose(
                predict(bundle.models[name], features),
                reference[f"{name}_risk_score"],
                rtol=0,
                atol=1e-12,
            )
        manifest.update(
            validation_rows=len(reference),
            reference_score_replay=True,
            reference_threshold=report_meta["threshold"],
            forest_parameters=parameters,
        )
        prep = json.loads(
            (settings["prepared"] / "metadata.json").read_text(encoding="utf-8")
        )
        # Return unused Arrow/report buffers before applying the same RAM guard.
        # This changes allocation lifetime only, never the estimate or model inputs.
        release_candidate_memory()
        resources = memory_preflight(
            prep["splits"]["train"]["rows"],
            prep["splits"]["train"]["legitimate"],
            1 / 200,
        )
        manifest["memory_preflight"] = resources
        progress(
            f"Memory preflight passed: {resources['estimated_array_bytes'] / 1024**3:.2f} GiB estimated working arrays"
        )
        progress(
            "Loading original training split; preprocessing is reused without refitting"
        )
        training, labels, train_ids = load_prepared_split(settings["prepared"], "train")
        verify_training_ids(train_ids, reference.source_row_number)
        values = feature_matrix(training)
        labels = validate_labels(labels, len(values), 5)
        if bundle.preprocessor.state.training_rows != len(values):
            raise ValueError("Preprocessor training row count mismatch")
        del training, train_ids
        release_candidate_memory()
        # Fail before any fit if one of the authorized ratios would not oversample.
        counts = np.bincount(labels, minlength=2)
        if any(
            int(counts[0] / d - counts[1]) <= 0 for d in settings["ratio_denominators"]
        ):
            raise ValueError(
                "An approved ratio is below the original minority proportion"
            )
        fixed_tables, selected_tables = [], []
        for denominator in settings["ratio_denominators"]:
            candidate = output / f"ratio_{denominator}"
            candidate.mkdir()
            config = bundle.config.model_copy(deep=True)
            config.smote.sampling_ratio = 1 / denominator
            manifest["active_ratio"] = denominator
            manifest["new_fits_started"] += 1
            write_json(output / "manifest.json", manifest)
            model, model_meta, audit, fitted = train_one(
                candidate / "model",
                values,
                labels,
                config,
                bundle,
                parameters,
                settings["prepared"],
                progress,
            )
            scores = predict(model, features)
            np.testing.assert_allclose(
                scores, predict(fitted, features), rtol=0, atol=1e-12
            )
            del fitted
            write_json(
                candidate / "validation_reload_verification.json",
                {
                    "passed": True,
                    "rows": len(scores),
                    "atol": 1e-12,
                    "all_validation_scores_replayed": True,
                },
            )
            fixed, selected, cutoff = analyze_pair(
                candidate,
                denominator,
                reference,
                scores,
                features,
                bundle,
                model,
                model_meta,
                settings,
                policy,
                root,
                progress,
            )
            fixed_tables.append(fixed)
            selected_tables.append(selected)
            manifest["completed_ratios"].append(
                {
                    "denominator": denominator,
                    "cutoff": cutoff,
                    "synthetic_rows": audit["synthetic_rows"],
                }
            )
            write_json(output / "manifest.json", manifest)
            del model, scores
            release_candidate_memory()
        del values, labels
        save_summary(
            output,
            pd.concat(fixed_tables, ignore_index=True),
            pd.concat(selected_tables, ignore_index=True),
            settings["ratio_denominators"],
        )
        progress(
            "Verifying all source artifacts and implementation fingerprints remain unchanged"
        )
        verify_unchanged(before)
        verify_unchanged(code_before)
        progress(
            "Complete: three fits, three verified comparisons, offline reports, source preservation"
        )
        manifest.update(
            status="complete",
            completed_at=now(),
            input_artifacts_unchanged=True,
            implementation_unchanged=True,
            offline_links_verified=True,
        )
        manifest.pop("active_ratio", None)
        manifest["files"] = {
            p.relative_to(output).as_posix(): file_sha256(p)
            for p in sorted(output.rglob("*"))
            if p.is_file() and p != output / "manifest.json"
        }
        write_json(output / "manifest.json", manifest)
        return output
    except BaseException as exc:
        manifest.update(
            status="failed", failed_at=now(), error=f"{type(exc).__name__}: {exc}"
        )
        progress(f"FAILED: {manifest['error']}")
        write_json(output / "manifest.json", manifest)
        (output / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        (output / "report.html").write_text(
            page(
                "Incomplete experiment",
                '<h1>Experiment incomplete</h1><p>Do not treat partial results as a completed comparison. See <a href="manifest.json">manifest</a> and <a href="run.log">log</a>.</p>',
            ),
            encoding="utf-8",
        )
        raise


def main(root):
    parser = argparse.ArgumentParser(
        description="Run three isolated ordinary-SMOTE ratio comparisons"
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
        )
        return 2
    print(
        json.dumps(
            {
                "success": True,
                "report": str(output / "report.html"),
                "manifest": str(output / "manifest.json"),
            }
        )
    )
    return 0
