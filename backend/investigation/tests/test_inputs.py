from types import SimpleNamespace

import pytest
import yaml

from diagnostics.inputs import (
    align_predictions,
    load_inputs,
    read_config,
    verify_links,
    verify_unchanged,
)
from integritree.ml.data import file_sha256


def align(frame, original):
    return align_predictions(
        frame, original.source_row_number, original.actual_label, "hash", "model", 0.43
    )


def test_align_accepts_reordering(frame):
    result = align(frame.iloc[::-1], frame)
    assert result.source_row_number.tolist() == frame.source_row_number.tolist()


@pytest.mark.parametrize(
    "change",
    ["missing", "duplicate", "labels", "test", "scores", "identity", "run", "null"],
)
def test_alignment_rejects_bad_evidence(frame, change):
    bad = frame.copy()
    if change == "missing":
        bad = bad.iloc[:-1]
    elif change == "duplicate":
        bad.loc[1, "source_row_number"] = 1
    elif change == "labels":
        bad.loc[0, "actual_label"] = 1
    elif change == "test":
        bad.loc[0, "split"] = "test"
    elif change == "scores":
        bad.loc[0, "rf_risk_score"] = 0.43
    elif change == "identity":
        bad.loc[0, "transaction_id"] = "other:1"
    elif change == "run":
        bad.loc[0, "run_id"] = "other"
    else:
        bad.loc[0, "transaction_id"] = None
    with pytest.raises(ValueError):
        align(bad, frame)


def test_provenance_mismatch_and_changed_artifact(tmp_path):
    model = tmp_path / "model"
    prep = tmp_path / "prep"
    model.mkdir()
    prep.mkdir()
    for path in (
        model / "metadata.json",
        prep / "metadata.json",
        prep / "split_manifest.parquet",
    ):
        path.write_text("example")
    bundle = SimpleNamespace(
        config=SimpleNamespace(
            dataset=SimpleNamespace(sha256="hash"),
            scoring=SimpleNamespace(threshold=0.43),
        ),
        metadata={"run_id": "model"},
    )
    meta = {
        "split": "validation",
        "official_test": False,
        "model_metadata_sha256": file_sha256(model / "metadata.json"),
        "prepared_metadata_sha256": file_sha256(prep / "metadata.json"),
        "split_manifest_sha256": file_sha256(prep / "split_manifest.parquet"),
        "model_run_id": "model",
        "dataset_sha256": "hash",
        "threshold": 0.43,
    }
    verify_links(meta, bundle, model, prep)
    with pytest.raises(ValueError, match="provenance"):
        verify_links(meta | {"dataset_sha256": "different"}, bundle, model, prep)
    before = {str(model / "metadata.json"): file_sha256(model / "metadata.json")}
    (model / "metadata.json").write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        verify_unchanged(before)


def test_test_report_rejected_before_model_loading(monkeypatch, tmp_path, frame):
    monkeypatch.setattr(
        "diagnostics.inputs.load_report",
        lambda _: ({"split": "test", "official_test": True}, frame),
    )

    def forbidden(*args):
        pytest.fail("Model loader must not be called for a test report")

    monkeypatch.setattr("diagnostics.inputs.load_bundle", forbidden)
    with pytest.raises(ValueError, match="validation reports only"):
        load_inputs(
            dict(models=tmp_path, prepared=tmp_path, report=tmp_path), lambda _: None
        )


def test_config_relative_to_its_directory_not_working_directory(tmp_path, monkeypatch):
    folder = tmp_path / "nested"
    folder.mkdir()
    path = folder / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            dict(
                schema_version=1,
                models="../models",
                prepared="../prepared",
                report="../report",
                sample_per_group=5,
                seed=42,
                minimum_denominator=20,
            )
        )
    )
    monkeypatch.chdir(tmp_path.parent)
    assert read_config(path)["models"] == tmp_path / "models"
    value = yaml.safe_load(path.read_text())
    value["sample_per_group"] = 0
    path.write_text(yaml.safe_dump(value))
    with pytest.raises(ValueError, match="sample_per_group"):
        read_config(path)
