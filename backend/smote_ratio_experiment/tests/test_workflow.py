"""Real tiny sampling, training, reload, SHAP and HTML; no real data are used."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
import yaml

from integritree.config import load_experiment
from integritree.ml.artifacts import ModelBundle, load_bundle
from integritree.ml.data import RAW_COLUMNS, file_sha256
from integritree.ml.preparation import prepare_dataset, load_prepared_split
from integritree.ml.preprocessing import FittedPreprocessor
from integritree.ml.training import forest_parameters
from ratio_experiment.core import load_model, predict
from ratio_experiment import runner

ROOT = Path(__file__).resolve().parents[1]


def test_real_three_ratio_workflow(tmp_path, monkeypatch):
    raw = pd.DataFrame(
        [
            {
                "step": i % 744 + 1,
                "type": "TRANSFER" if i % 2 else "CASH_OUT",
                "amount": float(i + 1),
                "nameOrig": f"C_{i}",
                "nameDest": "C_dest",
                "oldbalanceOrg": 10.0,
                "newbalanceOrig": 0.0,
                "oldbalanceDest": 0.0,
                "newbalanceDest": 10.0,
                "isFraud": int(i % 1000 == 0),
                "isFlaggedFraud": 0,
            }
            for i in range(20000)
        ],
        columns=RAW_COLUMNS,
    )
    source = tmp_path / "tiny.csv"
    raw.to_csv(source, index=False)
    config = load_experiment(ROOT.parent / "configs" / "experiment.yaml")
    config.dataset.filename = source.name
    config.dataset.sha256 = file_sha256(source)
    config.dataset.row_count = len(raw)
    config.random_forest.max_depth = 10
    config.smote.sampling_ratio = 0.01
    config.scoring.threshold = 0.43
    prepared = prepare_dataset(
        source, config, tmp_path / "data", "tiny", progress=lambda _: None
    )
    train, labels, _ = load_prepared_split(prepared, "train")
    features, truth, ids = load_prepared_split(prepared, "validation")
    features.index = pd.Index(ids, name="source_row_number")
    parameters = forest_parameters(config, jobs=1)
    rf = RandomForestClassifier(**parameters).fit(train.to_numpy(), labels)
    smote = RandomForestClassifier(**parameters).fit(train.to_numpy(), labels)
    metadata = {
        "prepared_metadata_sha256": file_sha256(prepared / "metadata.json"),
        "run_id": "fixture",
    }
    bundle = ModelBundle(
        {"rf": rf, "rf_smote": smote},
        FittedPreprocessor.load(prepared / "preprocessing.json"),
        config,
        metadata,
    )
    reference = raw.iloc[np.asarray(ids) - 1][["step", "type", "amount"]].reset_index(
        drop=True
    )
    reference["source_row_number"] = ids.to_numpy()
    reference["transaction_id"] = [f"{config.dataset.sha256}:{int(i)}" for i in ids]
    reference["actual_label"] = truth.to_numpy()
    reference["rf_risk_score"] = predict(rf, features)
    reference["rf_smote_risk_score"] = predict(smote, features)
    for name in ("is_zero_amount", "is_merchant_origin", "is_merchant_dest"):
        reference[name] = features[name].to_numpy()
    reference["split"] = "validation"
    reference["threshold"] = 0.43
    reference["run_id"] = "fixture"
    report = tmp_path / "reference"
    report.mkdir()
    (report / "metadata.json").write_text(
        json.dumps({"split": "validation", "official_test": False})
    )
    settings = yaml.safe_load((ROOT / "config.yaml").read_text())
    settings.update(
        prepared=str(prepared),
        report=str(report),
        models=str(report),
        sample_per_group=1,
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(settings))
    root = tmp_path / "experiment"
    root.mkdir()
    (root / "run.py").write_text("# fixture entry point")
    before = {
        str(p): file_sha256(p)
        for p in prepared.iterdir()
        if p.name != "test_features.parquet"
    }
    monkeypatch.setattr(runner, "input_inventory", lambda settings: before)
    monkeypatch.setattr(
        runner,
        "load_inputs",
        lambda settings, progress: (
            bundle,
            reference,
            features,
            config.evaluation,
            {"threshold": 0.43},
        ),
    )
    requested = []
    real_loader = runner.load_prepared_split

    def guarded_load(path, split):
        requested.append(split)
        assert split == "train"
        return real_loader(path, split)

    monkeypatch.setattr(runner, "load_prepared_split", guarded_load)
    output = runner.run(root, config_path, "smoke")
    assert requested == ["train"]
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["new_fits_started"] == 3
    assert [r["denominator"] for r in manifest["completed_ratios"]] == [200, 300, 500]
    assert manifest["input_artifacts_unchanged"] and not manifest["test_used"]
    for path, expected in before.items():
        assert file_sha256(Path(path)) == expected
    for name, expected in manifest["files"].items():
        assert file_sha256(output / name) == expected
    for d in (200, 300, 500):
        candidate = output / f"ratio_{d}"
        audit = json.loads((candidate / "model" / "training_audit.json").read_text())
        assert audit["original_rows_preserved"] and audit["original_labels_preserved"]
        assert audit["class_counts"]["1"] == int(
            audit["original_class_counts"]["0"] / d
        )
        model = load_model(candidate / "model", metadata["prepared_metadata_sha256"])
        prediction = pd.read_parquet(candidate / "predictions_selected.parquet")
        assert prediction.threshold.nunique() == 1
        np.testing.assert_array_equal(
            prediction.source_row_number, reference.source_row_number
        )
        np.testing.assert_allclose(
            predict(model, features), prediction.rf_smote_risk_score, rtol=0, atol=1e-12
        )
        with pytest.raises(ValueError):
            load_bundle(candidate / "model")
        with pytest.raises(ValueError, match="provenance"):
            load_model(candidate / "model", "wrong-hash")
        contributions = pd.read_csv(candidate / "tables" / "shap_contributions.csv")
        for _, group in contributions.groupby(["source_row_number", "model"]):
            assert (
                abs(
                    group.base_value.iloc[0]
                    + group.contribution.sum()
                    - group.output_value.iloc[0]
                )
                <= 1e-6
            )
    altered = output / "ratio_200" / "model" / "training_audit.json"
    altered.write_text(altered.read_text() + " ")
    with pytest.raises(ValueError, match="fingerprint"):
        load_model(altered.parent)
