"""Explicit, recoverable Stage 3 reset that preserves fitted candidate evidence."""
import argparse
from datetime import datetime, timezone
from pathlib import Path

from integritree.config import ExperimentConfig
from integritree.ml import staged_selection as selection
from integritree.ml.data import file_sha256


def contained(base, path):
    base, path = Path(base).resolve(), Path(path).resolve()
    if path == base or not path.is_relative_to(base):
        raise ValueError(f"Reset path escapes its allowed directory: {path}")
    return path


def verify_preserved(base, files):
    for name, expected in files.items():
        path = contained(base, base / name)
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"Preserved evidence fingerprint mismatch: {name}")


def _verify_candidates(run, plan, protocol):
    config = ExperimentConfig.model_validate(plan["configuration"])
    rows = selection.read_json(run / "search_progress.json")["completed_candidates"]
    ratio = selection.read_json(run / "stages/01_smote_ratio/decision.json")
    forest = selection.read_json(run / "stages/02_random_forest/decision.json")
    ratio_rows = [rows[selection.candidate_key(r, protocol.reference_forest)] for r in protocol.ratios]
    ratio_winner = selection.choose_ratio(ratio_rows)
    forest_rows = [rows[selection.candidate_key(ratio_winner["ratio"], f)] for f in protocol.forests()]
    forest_winner = selection.choose_forest(forest_rows)
    if set(rows) != {row["key"] for row in ratio_rows + forest_rows}:
        raise ValueError("Reset requires all Stage 1-2 candidates")
    for decision, candidates, winner, objective in (
        (ratio, ratio_rows, ratio_winner, protocol.ratio_objective),
        (forest, forest_rows, forest_winner, protocol.forest_objective),
    ):
        if decision != {"frozen": True, "objective": objective, "candidates": candidates,
                        "chosen_key": winner["key"]}:
            raise ValueError("Frozen Stage 1-2 decision differs")
    identity = None
    for row in rows.values():
        model = selection.resolve_reference(row["model"], run)
        report = selection.resolve_reference(row["report"], run)
        # Candidate references must remain within this run for an in-place reset.
        contained(run / "candidates", model)
        contained(run / "validation", report)
        _, actual = selection.inspect_model(model, plan["prepared_metadata_sha256"])
        requested = selection.candidate_config(config, row["ratio"], row["forest"])
        if selection.training_identity(actual) != selection.training_identity(requested):
            raise ValueError("Candidate configuration differs")
        predictions = selection.verify_report(report, model, config, plan["prepared_metadata_sha256"])
        current = predictions[["transaction_id", "source_row_number", "actual_label", "split"]]
        if identity is None:
            identity = current.copy()
        elif not identity.equals(current):
            raise ValueError("Candidate validation records differ")
        expected = selection.summarize_candidate(predictions, requested) | {
            "key": row["key"], "model": row["model"], "report": row["report"]}
        if expected != row:
            raise ValueError("Candidate metrics differ")
        del predictions, current
        selection.release_candidate_memory()


def _delete_file(base, name, expected):
    path = contained(base, base / name)
    if path.exists():
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"Cleanup target changed since inventory: {name}")
        path.unlink()


def reset_stage3(run, backend, protocol):
    """Remove only inventoried outputs; retry this function to finish a partial reset."""
    backend = Path(backend).resolve()
    run = contained(backend / "artifacts", run)
    if run.parent != backend / "artifacts":
        raise ValueError("Reset requires a direct artifacts child")
    review = contained(backend / "reports", backend / "reports" / run.name)
    runtime = backend / "runtime/validation"
    runtime.mkdir(parents=True, exist_ok=True)
    journal = runtime / f"{run.name}_percent_grid_reset.json"
    protocol = selection.SelectionProtocol.model_validate(protocol.model_dump())
    with selection.run_lock(run), selection.run_lock(run, ".control.lock"):
        if journal.exists():
            audit = selection.read_json(journal)
            if audit["run"] != str(run) or audit["new_plan"]["protocol"] != protocol.model_dump(mode="json"):
                raise ValueError("Reset journal belongs to a different run or protocol")
            verify_preserved(backend, audit["preserved_files"])
            if audit["status"] == "complete":
                return journal
        else:
            manifest = selection.read_json(run / "metadata.json")
            if (manifest.get("kind") != "validation_selection" or manifest.get("schema_version") != 2
                    or manifest.get("completed_stage") not in (2, 3)
                    or manifest.get("status") not in ("complete", "awaiting_next_stage")
                    or manifest.get("test_used") is not False):
                raise ValueError("Reset requires completed Stage 2 or Stage 3")
            plan = selection.read_json(run / "search_plan.json")
            if file_sha256(run / "search_plan.json") != manifest["search_plan_sha256"]:
                raise ValueError("Search plan fingerprint mismatch")
            for name, expected in (manifest.get("files", {}) | manifest["frozen_stage_files"]).items():
                if file_sha256(contained(run, run / name)) != expected:
                    raise ValueError("Existing evidence fingerprint mismatch")
            changed = plan["protocol"] | {"threshold_candidates": protocol.threshold_candidates}
            if changed != protocol.model_dump(mode="json"):
                raise ValueError("Only the threshold candidate policy may change")
            _verify_candidates(run, plan, protocol)
            preserved = [run / "search_progress.json"]
            for folder in ("candidates", "validation", "stages/01_smote_ratio", "stages/02_random_forest"):
                preserved.extend(p for p in (run / folder).rglob("*") if p.is_file())
            for stage in ("stage1_review", "stage2_review"):
                preserved.extend(p for name in ("candidates.csv", "decision.json")
                                 if (p := review / stage / name).is_file())
            targets = [run / name for name in ("selection.json", "SUMMARY.md")]
            targets += [review / name for name in ("selection.json", "SUMMARY.md", "metadata.json", "search_plan.json")]
            folders = [run / "stages/03_threshold", review / "stages/03_threshold", review / "final_validation"]
            for folder in folders:
                contained(backend, folder)
                targets.extend(p for p in folder.rglob("*") if p.is_file())
            current = runtime / "current_run.json"
            if current.exists():
                import json
                live = json.loads(current.read_text(encoding="utf-8-sig"))
                if live.get("run_id") == run.name and live.get("stop_after_stage") == 3:
                    stdout = contained(runtime, Path(live["stdout"]))
                    suffix = ".stdout.log"
                    if not stdout.name.endswith(suffix):
                        raise ValueError("Unexpected execution log name")
                    prefix = stdout.name[:-len(suffix)] + "."
                    targets.extend(p for p in runtime.iterdir() if p.is_file() and p.name.startswith(prefix))
                    targets.append(current)
            new_plan = plan | {"protocol": changed}
            new_plan["implementation_sha256"] = {
                name: file_sha256(Path(selection.__file__).with_name(name))
                for name in plan["implementation_sha256"]}
            clean = {k: v for k, v in manifest.items() if k not in (
                "files", "worker_pid", "failure", "stopped_at", "paused_at", "pause_requested_at")}
            clean.update(status="resetting_stage3", completed_stage=2, stage="forest_selected",
                         active_stage=2, active_candidate=None)
            clean["execution_history"] = [e for e in clean.get("execution_history", []) if e["stop_after_stage"] <= 2]
            audit = {"status": "pending", "run": str(run), "requested_at": datetime.now(timezone.utc).isoformat(),
                     "note": "Threshold protocol revised after validation inspection; candidate evidence preserved.",
                     "new_plan": new_plan, "reset_manifest": clean,
                     "preserved_files": {str(contained(backend, p).relative_to(backend)): file_sha256(p) for p in preserved},
                     "delete_files": {str(contained(backend, p).relative_to(backend)): file_sha256(p) for p in targets if p.is_file()},
                     "delete_directories": [str(p.relative_to(backend)) for folder in folders
                         for p in ([*folder.rglob("*"), folder] if folder.exists() else []) if p.is_dir()]}
            selection.atomic_json(journal, audit)
        # Mark incomplete before any deletion; a retry uses the same checked inventory.
        selection.atomic_json(run / "metadata.json", audit["reset_manifest"])
        for name, expected in audit["delete_files"].items():
            _delete_file(backend, name, expected)
        for name in sorted(audit["delete_directories"], key=len, reverse=True):
            path = contained(backend, backend / name)
            if path.exists():
                path.rmdir()  # Unexpected files cause a failure rather than broad deletion.
        selection.atomic_json(run / "search_plan.json", audit["new_plan"])
        manifest = audit["reset_manifest"] | {"search_plan_sha256": file_sha256(run / "search_plan.json")}
        selection.atomic_json(run / "metadata.json", manifest)
        # Exports require an awaiting-stage manifest. The reset journal guards resume until complete.
        manifest["status"] = "awaiting_next_stage"
        selection.atomic_json(run / "metadata.json", manifest)
        for export in (selection.export_ratio_stage, selection.export_forest_stage):
            folder = export(run, backend / "reports")
            summary = folder / "SUMMARY.md"
            summary.write_text(summary.read_text(encoding="utf-8") +
                               "\nReview regenerated under the revised threshold protocol; Stage 1-2 evidence is unchanged.\n", encoding="utf-8")
            meta = selection.read_json(folder / "metadata.json")
            meta["files"]["SUMMARY.md"] = file_sha256(summary)
            meta["regenerated_at"] = datetime.now(timezone.utc).isoformat()
            selection.atomic_json(folder / "metadata.json", meta)
        verify_preserved(backend, audit["preserved_files"])
        audit.update(status="complete", completed_at=datetime.now(timezone.utc).isoformat())
        selection.atomic_json(journal, audit)
        return journal


def main():
    from integritree.settings import load_settings
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=Path("configs/validation_three_stage.yaml"))
    args = parser.parse_args()
    base = load_settings().backend_root
    run = args.run if args.run.is_absolute() else base / args.run
    protocol = args.protocol if args.protocol.is_absolute() else base / args.protocol
    print(reset_stage3(run, base, selection.load_protocol(protocol)))
