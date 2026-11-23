"""Read-only model reuse for sequential validation on an engineered CSV."""
import argparse
import csv
from datetime import datetime, timezone
from fractions import Fraction
import gc
import html
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import sys
import time
import traceback

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(REPO / "backend/src"))
os.environ["MPLCONFIGDIR"] = str(ROOT / "runtime/matplotlib")

import numpy as np
import pandas as pd
from integritree.config import ExperimentConfig
from integritree.ml.artifacts import PACKAGES, load_bundle
from integritree.ml.data import file_sha256, write_json
from integritree.ml.evaluation import metrics
from integritree.ml.features import FEATURE_COLUMNS
from integritree.ml.inference import feature_matrix, fraud_scores
from integritree.ml.staged_selection import (
    SelectionProtocol, candidate_config, candidate_key, choose_forest, choose_ratio,
    inspect_model, resolve_reference, save_threshold_figure, training_identity,
)
from integritree.ml.threshold_search import percent_grid_thresholds

DEFAULT_SOURCE = REPO / "backend/artifacts/paysim_three_stage_20260928_172539"
DEFAULT_CSV = Path(r"C:\Users\Carpicorn\Downloads\Experiment-Dataset.csv")
KINDS = ("rf", "rf_smote")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle), [])
    if len(header) != len(set(header)):
        raise ValueError("Duplicate CSV column names")
    if set(header) != set(FEATURE_COLUMNS + ["isFraud"]):
        raise ValueError("Expected exactly the 11 engineered features and isFraud")
    frame = pd.read_csv(path, encoding="utf-8-sig", float_precision="round_trip")
    values = feature_matrix(frame[FEATURE_COLUMNS])
    labels = pd.to_numeric(frame.isFraud, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(labels).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError("isFraud must contain binary labels 0 and 1")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Both label classes are required")
    return values, labels.astype("uint8")


class SavedCandidates:
    """Inventory only the candidates referenced by the original saved run."""

    def __init__(self, source, tracked):
        self.source = source.resolve()
        self.tracked = tracked
        meta = read_json(source / "metadata.json")
        if (meta.get("kind"), meta.get("status"), meta.get("schema_version")) != (
            "validation_selection", "complete", 2
        ):
            raise ValueError("Source must be a completed three-stage selection run")
        self.track(source / "metadata.json")
        for name in ("search_plan.json", "search_progress.json"):
            self.track(source / name, meta["files"][name])
        plan = read_json(source / "search_plan.json")
        self.protocol = SelectionProtocol.model_validate(plan["protocol"])
        if self.protocol != SelectionProtocol():
            raise ValueError("Source must use the original eight-ratio, twelve-forest protocol")
        self.base = ExperimentConfig.model_validate(plan["configuration"])
        self.prepared_hash = plan["prepared_metadata_sha256"]
        self.entries = read_json(source / "search_progress.json")["completed_candidates"]
        self.verified = {}
        self.preprocessing_hash = None

    def track(self, path, expected=None):
        path = path.resolve()
        digest = file_sha256(path)
        if expected is not None and digest != expected:
            raise ValueError(f"Source fingerprint mismatch: {path}")
        previous = self.tracked.setdefault(str(path), digest)
        if previous != digest:
            raise ValueError(f"Source changed during run: {path}")

    def missing(self, ratio, forests):
        missing = []
        for forest in forests:
            key = candidate_key(ratio, forest)
            entry = self.entries.get(key)
            if entry is None:
                missing.append({"key": key, "ratio": ratio, "forest": list(forest)})
                continue
            path = (self.source / entry["model"]["path"]).resolve()
            if not all((path / name).is_file() for name in ("metadata.json", "rf.joblib", "rf_smote.joblib")):
                missing.append({"key": key, "ratio": ratio, "forest": list(forest)})
        return missing

    def get(self, ratio, forest):
        key = candidate_key(ratio, forest)
        if key in self.verified:
            return self.verified[key]
        entry = self.entries[key]
        if entry["ratio"] != ratio or tuple(entry["forest"]) != tuple(forest):
            raise ValueError("Candidate index does not match requested settings")
        path = resolve_reference(entry["model"], self.source)
        meta, config = inspect_model(path, self.prepared_hash)
        expected = candidate_config(self.base, ratio, forest)
        if training_identity(config) != training_identity(expected):
            raise ValueError("Saved training configuration does not match candidate")
        preprocessing_hash = meta["files"]["preprocessing.json"]
        if self.preprocessing_hash not in (None, preprocessing_hash):
            raise ValueError("Candidates use different preprocessing")
        self.preprocessing_hash = preprocessing_hash
        self.track(path / "metadata.json", entry["model"]["metadata_sha256"])
        for name, digest in meta["files"].items():
            self.track(path / name, digest)
        self.verified[key] = path, meta, config
        return self.verified[key]


def metric_row(labels, scores, threshold, policy):
    result = metrics(labels, scores, threshold, policy)
    row = {name: item["value"] for name, item in result["metrics"].items()}
    row.update({name + "_reason": item["reason"] for name, item in result["metrics"].items()})
    row.update({name: result["confusion_matrix"][name] for name in ("tp", "fp", "tn", "fn")})
    return row


def save_stage(output, stage, rows, winner, objective):
    folder = output / stage
    folder.mkdir()
    flattened = []
    for row in rows:
        item = {"key": row["key"], "ratio": row["ratio"], "trees": row["forest"][0],
                "depth": row["forest"][1], "min_samples_leaf": row["forest"][2],
                "mean_ap": row["mean_ap"], "selected": row["key"] == winner["key"]}
        for kind in KINDS:
            item.update({f"{kind}_{key}": value for key, value in row[kind].items()})
        flattened.append(item)
    pd.DataFrame(flattened).to_csv(folder / "candidates.csv", index=False, float_format="%.17g")
    write_json(folder / "decision.json", {"objective": objective, "winner": winner, "candidates": rows})


def verify_grid(labels, scores, curve, decision):
    """Independent direct classification and exact rational selection at all cutoffs."""
    if list(curve.threshold_percent) != list(range(1, 101)):
        raise ValueError("Threshold grid is incomplete")
    choices = []
    for row in curve.to_dict("records"):
        percent = int(row["threshold_percent"])
        threshold = percent / 100
        if row["threshold"] != threshold:
            raise ValueError("Threshold value differs from its percentage")
        f1 = []
        for kind in KINDS:
            flagged = scores[kind] >= threshold
            tp = int(np.count_nonzero(flagged & (labels == 1)))
            fp = int(np.count_nonzero(flagged & (labels == 0)))
            fn = int(np.count_nonzero(~flagged & (labels == 1)))
            tn = int(np.count_nonzero(~flagged & (labels == 0)))
            for name, count in dict(tp=tp, fp=fp, fn=fn, tn=tn).items():
                if row[f"{kind}_{name}"] != count:
                    raise ValueError("Threshold confusion count mismatch")
            f1.append(Fraction(2 * tp, 2 * tp + fp + fn))
        choices.append(((f1[0] + f1[1]) / 2, -abs(percent - 50), percent))
    best = max(choices)
    if decision["threshold"] != best[2] / 100 or Fraction(decision["mean_f1_fraction"]) != best[0]:
        raise ValueError("Selected cutoff does not match independent verification")
    return {"thresholds_verified": 100, "exact_selected_mean_f1": str(best[0])}


def report(output, manifest):
    title = "Experimental three-stage validation"
    note = ("Saved models only; zero training fits. Features are used as supplied. "
            "Selections and reported metrics use the same experimental validation CSV. "
            "Dataset independence and overlap have not been established. AP means Average Precision. "
            "Stage 1/2 classification metrics use 50%; their selection objective is AP. "
            "No application model or cutoff is changed.")
    lines = [f"# {title}", "", f"Status: **{manifest['status']}**", "", note, "",
             f"Rows: {manifest.get('rows', 'unavailable')}; fraud: {manifest.get('actual_fraud', 'unavailable')}.", ""]
    sections = []
    for stage, label in (("01_smote_ratio", "Stage 1: SMOTE ratio"), ("02_random_forest", "Stage 2: forest")):
        path = output / stage / "candidates.csv"
        if path.exists():
            frame = pd.read_csv(path, float_precision="round_trip")
            fields = ["ratio", "trees", "depth", "min_samples_leaf", "rf_pr_auc", "rf_smote_pr_auc", "mean_ap", "selected"]
            sections.append(f"<h2>{label}</h2>" + frame[fields].to_html(index=False, float_format=lambda x: f"{x:.9f}") +
                            f'<p><a href="{stage}/candidates.csv">All candidate metrics (CSV)</a> · <a href="{stage}/decision.json">Decision evidence</a></p>')
            winner = read_json(output / stage / "decision.json")["winner"]
            lines.extend([f"## {label}", "", f"Selected: `{winner['key']}`.", "", f"[All metrics]({stage}/candidates.csv) | [Decision]({stage}/decision.json)", ""])
    if "missing_candidates" in manifest:
        text = "Sequential selection stopped: the following saved candidates are unavailable. No alternative ratio was substituted."
        lines.extend([text, ""] + [f"- `{item['key']}`" for item in manifest["missing_candidates"]] + [""])
        sections.append("<h2>Missing saved models</h2><p>" + text + "</p><ul>" +
                        "".join("<li>" + html.escape(item["key"]) + "</li>" for item in manifest["missing_candidates"]) + "</ul>")
    if "selection" in manifest:
        choice = manifest["selection"]
        text = f"Selected ratio: {choice['ratio']:g}; forest (trees/depth/minimum leaf): {choice['forest']}; shared cutoff: {choice['threshold']:.0%}."
        lines.extend(["## Stage 3: shared cutoff", "", text, "", "[Final metrics](final_metrics.csv) | [All 100 cutoffs](03_threshold/thresholds.csv)", "", "![Threshold sweep](03_threshold/thresholds.svg)", ""])
        sections.append("<h2>Stage 3: shared cutoff</h2><p>" + html.escape(text) + "</p>" +
                        pd.read_csv(output / "final_metrics.csv").to_html(index=False) +
                        '<p><a href="final_metrics.csv">Final metrics</a> · <a href="03_threshold/thresholds.csv">All 100 cutoffs</a></p><img src="03_threshold/thresholds.svg" alt="F1 over all 100 cutoffs">')
    if "error" in manifest:
        lines.extend(["Failure: " + manifest["error"], ""])
        sections.append("<p>Failure: " + html.escape(manifest["error"]) + "</p>")
    lines.extend(["[Manifest](manifest.json) | [Verification](verification.json) | [Log](run.log)", ""])
    (output / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    document = ('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
                f'<title>{title}</title><style>body{{font:16px system-ui;max-width:1250px;margin:40px auto;padding:0 20px;color:#172e30}}'
                'table{border-collapse:collapse;display:block;overflow:auto;font-size:14px}th,td{padding:8px;border-bottom:1px solid #ddd;text-align:right}'
                'h1,h2{color:#146356}img{max-width:100%}a{color:#146356}</style>'
                f'<h1>{title}</h1><p>Status: <strong>{html.escape(manifest["status"])}</strong></p>'
                f'<p>{manifest.get("rows", "Unavailable")} rows; {manifest.get("actual_fraud", "unavailable")} fraud labels.</p><p>{note}</p>' +
                "".join(sections) + '<p><a href="manifest.json">Manifest</a> · <a href="verification.json">Verification</a> · <a href="run.log">Log</a></p></html>')
    (output / "report.html").write_text(document, encoding="utf-8")
    for target in re.findall(r'(?:href|src)="([^"]+)"', document):
        if not (output / target).is_file():
            raise ValueError(f"Broken report link: {target}")


def execute(csv_path, source, run_id=None, *, output_root=None):
    started = time.monotonic()
    run_id = run_id or datetime.now(timezone.utc).strftime("validation_%Y%m%dT%H%M%S_%fZ")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id):
        raise ValueError("Run ID must be a simple directory name")
    output = (output_root or ROOT / "outputs") / run_id
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"kind": "experimental_three_stage_validation", "schema_version": 1,
                "status": "running", "created_at": datetime.now(timezone.utc).isoformat(),
                "validation_csv": str(csv_path.resolve()), "source_run": str(source.resolve()),
                "training_fits": 0, "data_provenance_audited": False, "completed_stage": 0,
                "completed_candidates": [], "source_hashes": {},
                "package_versions": {p: version(p) for p in (*PACKAGES, "scipy", "matplotlib")}}
    tracked = manifest["source_hashes"]

    def log(message):
        message = datetime.now(timezone.utc).isoformat() + " " + message
        print(message, file=sys.stderr, flush=True)
        with (output / "run.log").open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")

    def checkpoint():
        manifest["elapsed_seconds"] = time.monotonic() - started
        write_json(output / "manifest.json", manifest)

    log("Starting validation-only replay; no training or resampling")
    checkpoint()
    verification = {}
    try:
        tracked[str(csv_path.resolve())] = file_sha256(csv_path)
        manifest["validation_csv_sha256"] = tracked[str(csv_path.resolve())]
        for path in list(ROOT.glob("*.py")) + list((REPO / "backend/src/integritree").rglob("*.py")):
            tracked[str(path.resolve())] = file_sha256(path)
        values, labels = load_csv(csv_path)
        manifest.update(rows=len(labels), actual_fraud=int(labels.sum()))
        inventory = SavedCandidates(source, tracked)
        protocol = inventory.protocol
        manifest["protocol"] = protocol.model_dump()
        rows_by_key, rf_cache = {}, {}
        (output / "candidates").mkdir()

        def evaluate(ratio, forest):
            key = candidate_key(ratio, forest)
            if key in rows_by_key:
                return rows_by_key[key]
            log(f"Scoring {key}")
            path, meta, config = inventory.get(ratio, forest)
            bundle = load_bundle(path)
            scores = {}
            digest = meta["files"]["rf.joblib"]
            if digest in rf_cache:
                scores["rf"] = np.load(rf_cache[digest], allow_pickle=False)
            else:
                scores["rf"] = fraud_scores(bundle.models["rf"], values)
            scores["rf_smote"] = fraud_scores(bundle.models["rf_smote"], values)
            del bundle
            gc.collect()
            folder = output / "candidates" / key
            folder.mkdir()
            if digest not in rf_cache:
                np.save(folder / "rf_scores.npy", scores["rf"], allow_pickle=False)
                rf_cache[digest] = folder / "rf_scores.npy"
            predictions = pd.DataFrame({"source_row_number": np.arange(1, len(labels) + 1), "actual_label": labels,
                                        **{f"{kind}_risk_score": scores[kind] for kind in KINDS}})
            predictions.to_parquet(folder / "predictions.parquet", index=False)
            row = {"key": key, "ratio": ratio, "forest": list(forest), "model_path": str(path),
                   "model_metadata_sha256": file_sha256(path / "metadata.json"), "reference_threshold": 0.5}
            for kind in KINDS:
                row[kind] = metric_row(labels, scores[kind], 0.5, inventory.base.evaluation)
            row["mean_ap"] = (row["rf"]["pr_auc"] + row["rf_smote"]["pr_auc"]) / 2
            write_json(folder / "metrics.json", row)
            rows_by_key[key] = row
            manifest["completed_candidates"].append(key)
            checkpoint()
            log(f"Completed {key}: RF AP={row['rf']['pr_auc']:.9f}; RF-SMOTE AP={row['rf_smote']['pr_auc']:.9f}")
            return row

        missing = [item for ratio in protocol.ratios for item in inventory.missing(ratio, [protocol.reference_forest])]
        if missing:
            manifest.update(status="incomplete_missing_models", missing_candidates=missing)
        else:
            ratio_rows = [evaluate(ratio, protocol.reference_forest) for ratio in protocol.ratios]
            ratio_winner = choose_ratio(ratio_rows)
            save_stage(output, "01_smote_ratio", ratio_rows, ratio_winner, protocol.ratio_objective)
            manifest["completed_stage"] = 1
            ratio = ratio_winner["ratio"]
            log(f"Stage 1 selected ratio {ratio:g}")
            checkpoint()
            missing = inventory.missing(ratio, protocol.forests())
            if missing:
                manifest.update(status="incomplete_missing_models", missing_candidates=missing)
                log(f"Stopping: {len(missing)} required Stage 2 pairs unavailable")
            else:
                forest_rows = [evaluate(ratio, forest) for forest in protocol.forests()]
                winner = choose_forest(forest_rows)
                save_stage(output, "02_random_forest", forest_rows, winner, protocol.forest_objective)
                manifest["completed_stage"] = 2
                checkpoint()
                predictions = pd.read_parquet(output / "candidates" / winner["key"] / "predictions.parquet")
                scores = {kind: predictions[f"{kind}_risk_score"].to_numpy() for kind in KINDS}
                decision, curve = percent_grid_thresholds(labels, scores["rf"], scores["rf_smote"])
                folder = output / "03_threshold"
                folder.mkdir()
                curve.to_csv(folder / "thresholds.csv", index=False, float_format="%.17g")
                curve.to_parquet(folder / "thresholds.parquet", index=False)
                write_json(folder / "decision.json", decision)
                save_threshold_figure(curve, decision, folder / "thresholds.svg")
                verification.update(verify_grid(labels, scores, pd.read_parquet(folder / "thresholds.parquet"), decision))
                final_rows = []
                for kind in KINDS:
                    final_rows.append({"model": kind, "threshold": decision["threshold"],
                                       **metric_row(labels, scores[kind], decision["threshold"], inventory.base.evaluation)})
                    predictions[f"{kind}_predicted_label"] = (scores[kind] >= decision["threshold"]).astype("uint8")
                pd.DataFrame(final_rows).to_csv(output / "final_metrics.csv", index=False, float_format="%.17g")
                write_json(output / "final_metrics.json", {row["model"]: row for row in final_rows})
                predictions.to_parquet(output / "final_predictions.parquet", index=False)
                manifest.update(status="complete", completed_stage=3,
                                selection={"ratio": ratio, "forest": winner["forest"], "key": winner["key"], **decision})
                log(f"Stage 3 selected cutoff {decision['threshold']:.0%}")
        log("Verifying all input and model fingerprints remain unchanged")
        for name, digest in tracked.items():
            if file_sha256(Path(name)) != digest:
                raise ValueError(f"Source changed: {name}")
        verification.update(source_preservation_verified=True, training_fits=0)
        write_json(output / "verification.json", verification)
        report(output, manifest)
        verification["report_links_verified"] = True
        write_json(output / "verification.json", verification)
    except Exception as exc:
        manifest.update(status="failed", error=str(exc))
        log(traceback.format_exc())
        write_json(output / "verification.json", verification | {"error": str(exc)})
        report(output, manifest)
        raise
    finally:
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        log(f"Finished: {manifest['status']}")
        manifest["files"] = {str(p.relative_to(output)).replace("\\", "/"): file_sha256(p)
                             for p in output.rglob("*") if p.is_file() and p.name != "manifest.json"}
        checkpoint()
    return output, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    output, manifest = execute(args.csv.resolve(), args.source_run.resolve(), args.run_id)
    print(output)
    return 0 if manifest["status"] == "complete" else 3


if __name__ == "__main__":
    raise SystemExit(main())
