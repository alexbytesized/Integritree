from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import run


def csv_fixture(tmp_path):
    frame = pd.DataFrame(np.zeros((6, 11)), columns=run.FEATURE_COLUMNS)
    frame[run.FEATURE_COLUMNS[0]] = [0.1, 0.9, 0.2, 0.8, 0.1, 0.9]
    frame["isFraud"] = [0, 1, 0, 1, 0, 1]
    path = tmp_path / "input.csv"
    frame.to_csv(path, index=False)
    return path, frame


def test_csv_order_duplicates_labels(tmp_path):
    path, frame = csv_fixture(tmp_path)
    frame[list(reversed(frame.columns))].to_csv(path, index=False)
    values, labels = run.load_csv(path)
    assert values.shape == (6, 11)
    np.testing.assert_array_equal(values, frame[run.FEATURE_COLUMNS])
    np.testing.assert_array_equal(labels, frame.isFraud)
    np.testing.assert_array_equal(values[0], values[4])


@pytest.mark.parametrize("change", ["duplicate_header", "extra", "missing", "nan", "inf", "label", "single_class", "empty"])
def test_bad_csv(tmp_path, change):
    path, frame = csv_fixture(tmp_path)
    if change == "duplicate_header":
        path.write_text("isFraud,isFraud\n0,1\n")
    else:
        if change == "extra": frame["extra"] = 0
        if change == "missing": frame = frame.drop(columns=run.FEATURE_COLUMNS[0])
        if change == "nan": frame.iloc[0, 0] = np.nan
        if change == "inf": frame.iloc[0, 0] = np.inf
        if change == "label": frame.loc[0, "isFraud"] = 2
        if change == "single_class": frame["isFraud"] = 0
        if change == "empty": frame = frame.iloc[:0]
        frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        run.load_csv(path)


def test_selection_and_threshold_ties():
    rows = [{"ratio": r, "rf_smote": {"pr_auc": 0.8}} for r in [0.02, 0.01]]
    assert run.choose_ratio(rows)["ratio"] == 0.01
    rows = [{"forest": f, "mean_ap": 0.8} for f in run.SelectionProtocol().forests()]
    assert run.choose_forest(rows)["forest"] == (100, 10, 50)
    labels = np.array([1, 0, 0, 1])
    scores = {k: np.array([0.49, 0.5, 0.5, 0.51]) for k in run.KINDS}
    best, curve = run.percent_grid_thresholds(labels, *scores.values())
    assert best["threshold"] == 0.51
    assert run.verify_grid(labels, scores, curve, best)["thresholds_verified"] == 100
    curve.loc[0, "rf_tp"] += 1
    with pytest.raises(ValueError, match="confusion"):
        run.verify_grid(labels, scores, curve, best)


@pytest.mark.parametrize("missing", [False, True])
def test_pipeline_sequential_without_fitting(tmp_path, monkeypatch, missing):
    path, _ = csv_fixture(tmp_path)
    from integritree.config import load_experiment
    from integritree.ml import training
    from sklearn.ensemble import RandomForestClassifier
    from imblearn.over_sampling import SMOTE

    def forbidden(*args, **kwargs):
        raise AssertionError("Training/resampling forbidden")
    monkeypatch.setattr(training, "train_models", forbidden)
    monkeypatch.setattr(RandomForestClassifier, "fit", forbidden)
    monkeypatch.setattr(SMOTE, "fit_resample", forbidden)
    config = load_experiment(run.REPO / "backend/configs/experiment.yaml")
    model_dir = tmp_path / "saved"
    model_dir.mkdir()
    (model_dir / "metadata.json").write_text("{}")
    run.write_json(model_dir / "search_plan.json", {"configuration": config.model_dump()})
    calls = []

    class Inventory:
        protocol = run.SelectionProtocol()
        base = config

        def __init__(self, *args): pass

        def missing(self, ratio, forests):
            if missing and len(forests) == 12:
                assert ratio == 0.02
                return [{"key": run.candidate_key(ratio, f), "ratio": ratio, "forest": list(f)} for f in forests[1:]]
            return []

        def get(self, ratio, forest):
            calls.append((ratio, forest))
            self.current = ratio
            return model_dir, {"files": {"rf.joblib": "same-rf"}}, config

    inventory = Inventory()
    monkeypatch.setattr(run, "SavedCandidates", lambda *args: inventory)

    class Model:
        classes_ = np.array([0, 1])

        def __init__(self, reverse=False): self.reverse = reverse

        def predict_proba(self, values):
            assert values.shape == (6, 11)  # Label never enters inference.
            score = values[:, 0]
            if self.reverse: score = 1 - score
            return np.column_stack([1 - score, score])

    monkeypatch.setattr(run, "load_bundle", lambda _: SimpleNamespace(models={
        "rf": Model(), "rf_smote": Model(reverse=inventory.current != (0.02 if missing else 0.01))}))
    output, manifest = run.execute(path, model_dir, "fixture", output_root=tmp_path / "outputs")
    assert manifest["training_fits"] == 0
    assert (output / "report.html").is_file()
    assert manifest["completed_stage"] == (1 if missing else 3)
    assert manifest["status"] == ("incomplete_missing_models" if missing else "complete")
    assert len(calls) == (8 if missing else 19)
    from verify import verify
    assert verify(output)["candidates_verified"] == len(calls)
    if missing:
        assert not (output / "03_threshold").exists()
        assert len(manifest["missing_candidates"]) == 11
    else:
        assert manifest["selection"]["ratio"] == 0.01
        assert len(pd.read_csv(output / "03_threshold/thresholds.csv")) == 100
    with pytest.raises(FileExistsError):
        run.execute(path, model_dir, "fixture", output_root=tmp_path / "outputs")


def test_source_rejects_tampered_plan(tmp_path):
    run.write_json(tmp_path / "metadata.json", {
        "kind": "validation_selection", "status": "complete", "schema_version": 2,
        "files": {"search_plan.json": "invalid"}})
    (tmp_path / "search_plan.json").write_text("{}")
    with pytest.raises(ValueError, match="fingerprint"):
        run.SavedCandidates(tmp_path, {})


def test_failure_is_preserved(tmp_path):
    path, frame = csv_fixture(tmp_path)
    frame["isFraud"] = 0
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Both"):
        run.execute(path, tmp_path, "failure", output_root=tmp_path / "outputs")
    output = tmp_path / "outputs/failure"
    assert run.read_json(output / "manifest.json")["status"] == "failed"
    assert "Both label classes" in (output / "report.html").read_text()
