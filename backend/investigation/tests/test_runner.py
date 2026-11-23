import json
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

from diagnostics.runner import run, verify_explanation
from diagnostics.reporting import verify_offline_report
from integritree.ml.features import FEATURE_COLUMNS


def explanation(row, model):
    return dict(
        model=model,
        transaction_id=row.transaction_id,
        output_value=row[f"{model}_risk_score"],
        predicted_label=int(row[f"{model}_predicted_label"]),
        threshold=row.threshold,
        base_value=0.1,
        features=[
            dict(
                feature="log_amount",
                contribution=row[f"{model}_risk_score"] - 0.1,
                model_value=0.2,
                readable_value="Amount",
            )
        ],
        narrative="Example case.",
    )


@pytest.mark.parametrize("failure", [None, "score", "sum", "identity", "threshold"])
def test_shap_parity_and_reconstruction(frame, failure):
    row = frame.iloc[0]
    item = explanation(row, "rf")
    if failure == "score":
        item["output_value"] += 0.01
    if failure == "sum":
        item["features"][0]["contribution"] += 0.01
    if failure == "identity":
        item["transaction_id"] = "wrong"
    if failure == "threshold":
        item["threshold"] = 0.5
    if failure:
        with pytest.raises(ValueError):
            verify_explanation(item, row, "rf", 1e-6)
    else:
        verify_explanation(item, row, "rf", 1e-6)


def setup_root(tmp_path):
    root = tmp_path / "investigation"
    root.mkdir()
    (root / "run.py").write_text("# fixture")
    (root / "diagnostics").mkdir()
    report = root / "source_report"
    report.mkdir()
    (report / "metadata.json").write_text(
        json.dumps({"split": "validation", "official_test": False})
    )
    config_path = root / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            dict(
                schema_version=1,
                models="models",
                prepared="prepared",
                report="source_report",
                sample_per_group=5,
                seed=42,
                minimum_denominator=20,
            )
        )
    )
    return root, config_path, report


def test_completed_workflow_and_non_overwrite(tmp_path, monkeypatch, frame, config):
    root, path, report = setup_root(tmp_path)
    features = pd.DataFrame(
        0.0,
        index=pd.Index(frame.source_row_number, name="source_row_number"),
        columns=FEATURE_COLUMNS,
    )
    bundle = SimpleNamespace(config=config)
    monkeypatch.setattr("diagnostics.runner.input_inventory", lambda _: {})
    monkeypatch.setattr(
        "diagnostics.runner.load_inputs",
        lambda *args: (
            bundle,
            frame,
            features,
            config.evaluation,
            {"threshold": 0.43, "model_run_id": "model"},
        ),
    )

    class Engine:
        def __init__(self, *args, **kwargs):
            pass

        def explain(self, features, ids, analysis_id):
            row = frame.set_index("transaction_id", drop=False).loc[ids[0]]
            result = []
            for model in ("rf", "rf_smote"):
                item = explanation(row, model)
                svg = root / f"{model}.svg"
                svg.write_text(
                    '<svg xmlns="http://www.w3.org/2000/svg" width="200" height="80"><text x="5" y="30">SHAP fixture</text></svg>'
                )
                item["waterfall_path"] = str(svg)
                result.append(item)
            return result

    monkeypatch.setattr("diagnostics.runner.ExplanationEngine", Engine)
    output = run(root, path, "fixture")
    meta = json.loads((output / "manifest.json").read_text())
    assert meta["status"] == "complete" and meta["inputs_unchanged"]
    assert meta["sampled_transactions"] == 8 and meta["paired_explanations"] == 16
    assert meta["test_used"] is False
    verify_offline_report(output)
    assert len(pd.read_parquet(output / "validation_details.parquet")) == 8
    assert "Completed diagnostic report" in (output / "report.html").read_text(
        encoding="utf-8"
    )
    with pytest.raises(FileExistsError):
        run(root, path, "fixture")
    assert json.loads((output / "manifest.json").read_text())["status"] == "complete"


def test_failed_run_is_explicit_and_never_reads_test(monkeypatch, tmp_path):
    root, path, report = setup_root(tmp_path)
    (report / "metadata.json").write_text(
        json.dumps({"split": "test", "official_test": True})
    )

    def forbidden(*args):
        pytest.fail("Must reject test input before inventory")

    monkeypatch.setattr("diagnostics.runner.input_inventory", forbidden)
    with pytest.raises(ValueError, match="validation reports only"):
        run(root, path, "rejected")
    output = root / "outputs" / "rejected"
    assert json.loads((output / "manifest.json").read_text())["status"] == "failed"
    assert "Investigation incomplete" in (output / "report.html").read_text()


def test_offline_check_rejects_remote_and_missing_links(tmp_path):
    page = tmp_path / "report.html"
    for content in [
        '<img src="https://example.com/plot.svg">',
        '<a href="absent.csv">Data</a>',
    ]:
        page.write_text(content)
        with pytest.raises(ValueError):
            verify_offline_report(tmp_path)
