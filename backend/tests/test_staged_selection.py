"""Sequential protocol checks with synthetic data and explicit held-out guards."""
from fractions import Fraction
import json

import numpy as np
import pandas as pd
import pytest

from integritree.config import load_experiment
from integritree.settings import BACKEND_ROOT
from integritree.ml import staged_selection as staged
from integritree.ml.artifacts import load_bundle
from integritree.ml.data import RAW_COLUMNS, file_sha256
from integritree.ml.features import FEATURE_COLUMNS
from integritree.ml.preparation import prepare_dataset
from integritree.ml.selection import exact_mean_f1
from integritree.ml.threshold_search import all_score_thresholds
from integritree.ml.training import resample_training, memory_preflight


def test_declared_protocol_and_ties():
    protocol = staged.load_protocol(BACKEND_ROOT / "configs/validation_three_stage.yaml")
    assert protocol.ratios == [.1, .2, 1 / 3, .5, 1]
    assert protocol.reference_forest == (100, 10, 1)
    assert len(protocol.forests()) == 12
    assert set(protocol.forests()) == {(t, d, l) for t in (100, 200) for d in (10, 20) for l in (1, 10, 50)}
    ratios = [{"ratio": r, "rf_smote": {"pr_auc": .8}} for r in reversed(protocol.ratios)]
    assert staged.choose_ratio(ratios)["ratio"] == .1
    forests = [{"forest": f, "mean_ap": .8} for f in reversed(protocol.forests())]
    assert staged.choose_forest(forests)["forest"] == (100, 10, 50)
    ratios[-1]["rf_smote"]["pr_auc"] = .7
    assert staged.choose_ratio(ratios)["ratio"] == .2
    forests[0]["mean_ap"] = .9
    assert staged.choose_forest(forests) is forests[0]
    for change in ({"ratios": []}, {"ratios": [float("nan")]}, {"ratios": [0]},
                   {"leaves": [1, 1]}, {"selection_split": "test"}, {"trees": [0]}):
        with pytest.raises(ValueError):
            staged.SelectionProtocol.model_validate(protocol.model_dump() | change)


@pytest.mark.parametrize("seed", range(10))
def test_exact_threshold_sweep_matches_exhaustive(seed):
    rng = np.random.default_rng(seed)
    labels = np.r_[0, 1, rng.integers(0, 2, size=28)]
    scores = rng.choice([0, .125, .25, .5, .75, .96, .9876543210987654, 1.], size=(2, 30))
    best, curve = all_score_thresholds(labels, *scores)
    candidates = sorted(set(np.r_[scores.ravel(), 0, .5, 1]))
    expected = max(candidates, key=lambda t: (exact_mean_f1(labels, *scores, t),
                                             -abs(Fraction(float(t)) - Fraction(1, 2)), t))
    assert best["threshold"] == expected
    assert Fraction(best["mean_f1_fraction"]) == exact_mean_f1(labels, *scores, expected)
    assert best["candidate_count"] == len(candidates)
    assert list(curve.threshold) == candidates
    for row in curve.to_dict("records"):
        for name, values in zip(("rf", "rf_smote"), scores):
            predicted = values >= row["threshold"]
            for field, actual, prediction in (("tp", 1, True), ("fp", 0, True),
                                               ("tn", 0, False), ("fn", 1, False)):
                assert row[f"{name}_{field}"] == np.count_nonzero((labels == actual) & (predicted == prediction))
    repeated, same = all_score_thresholds(labels, *scores)
    assert repeated == best
    pd.testing.assert_frame_equal(curve, same, check_exact=True)


def test_high_threshold_and_closest_half_ties():
    scores = [.96, .97, .9876543210987654, .99]
    best, _ = all_score_thresholds([0, 0, 1, 1], scores, scores)
    assert best["threshold"] == scores[2] > .95
    assert all_score_thresholds([0, 1], [0, 1], [0, 1])[0]["threshold"] == .5
    # Equidistant binary-exact cutoffs: prefer the higher one.
    best, _ = all_score_thresholds([0, 0, 1, 1], [.5, .5, .25, .75], [.5, .5, .25, .75])
    assert best["threshold"] == .75
    with pytest.raises(ValueError, match="both"):
        all_score_thresholds([0, 0], [0, 1], [0, 1])


@pytest.mark.parametrize("ratio", [.1, .2, 1 / 3, .5, 1.])
def test_all_resampling_counts_and_memory(ratio):
    config = load_experiment(BACKEND_ROOT / "configs/experiment.yaml")
    config.smote.sampling_ratio = ratio
    values = np.random.default_rng(42).random((211, len(FEATURE_COLUMNS)))
    labels = np.r_[np.ones(10, dtype="uint8"), np.zeros(201, dtype="uint8")]
    original = values.copy()
    resampled, targets, audit = resample_training(values, labels, config)
    minority = int(201 * ratio)
    assert len(targets) == 201 + minority
    assert audit["class_counts"] == {"0": 201, "1": minority}
    assert audit["generated_fraud_rows"] == minority - 10
    assert audit["requested_sampling_ratio"] == ratio
    assert audit["achieved_sampling_ratio"] == minority / 201
    np.testing.assert_array_equal(resampled[:211], original)
    np.testing.assert_array_equal(targets[:211], labels)
    assert (targets[211:] == 1).all()
    preflight = memory_preflight(211, 201, ratio)
    assert preflight["estimated_resampled_rows"] == len(targets)


@pytest.fixture
def experiment(tmp_path, raw_record):
    frame = pd.DataFrame([raw_record | {"step": i + 1, "amount": float(i + 1),
             "nameOrig": f"C_{i}", "type": "TRANSFER" if i % 2 else "CASH_OUT",
             "isFraud": int(i % 25 == 0)} for i in range(500)]).loc[:, RAW_COLUMNS]
    source = tmp_path / "synthetic.csv"
    frame.to_csv(source, index=False)
    config = load_experiment(BACKEND_ROOT / "configs/experiment.yaml").model_copy(deep=True)
    config.dataset = config.dataset.model_copy(update={"filename": source.name,
                  "sha256": file_sha256(source), "row_count": len(frame)})
    prepared = prepare_dataset(source, config, tmp_path / "prepared", "tiny", progress=lambda _: None)
    protocol = staged.SelectionProtocol(trees=[3, 4], depths=[2, 3], leaves=[1, 2, 4], reference_forest=(3, 2, 1))
    return prepared, config, protocol


def test_resume_reuse_final_bundle_and_corruption(experiment, tmp_path, monkeypatch):
    prepared, config, protocol = experiment
    import integritree.ml.training as training
    import integritree.ml.research as research
    real_training_loader, real_eval_loader = training.load_prepared_split, research.load_prepared_split
    reads = []
    def training_only(path, split):
        assert split == "train"
        reads.append(split)
        return real_training_loader(path, split)
    def validation_only(path, split):
        assert split == "validation"
        reads.append(split)
        return real_eval_loader(path, split)
    monkeypatch.setattr(training, "load_prepared_split", training_only)
    monkeypatch.setattr(research, "load_prepared_split", validation_only)
    fitted = []
    actual_fit = training.RandomForestClassifier.fit
    def record_fit(self, X, y, *args, **kwargs):
        fitted.append((self.n_estimators, self.max_depth, self.min_samples_leaf, len(y)))
        return actual_fit(self, X, y, *args, **kwargs)
    monkeypatch.setattr(training.RandomForestClassifier, "fit", record_fit)
    actual_train = staged.train_models
    calls = []
    def interrupt_after_complete_fit(*args, **kwargs):
        calls.append(1)
        result = actual_train(*args, **kwargs)
        if len(calls) == 6:
            raise RuntimeError("synthetic interruption after saved fit")
        return result
    monkeypatch.setattr(staged, "train_models", interrupt_after_complete_fit)
    output = tmp_path / "selections" / "run"
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        staged.select_three_stage(prepared, config, protocol, output.parent, "run", progress=lambda _: None)
    assert (output / "stages/01_smote_ratio/decision.json").exists()
    assert not (output / "stages/02_random_forest/decision.json").exists()
    frozen_hash = file_sha256(output / "stages/01_smote_ratio/decision.json")
    assert len(staged.read_json(output / "search_progress.json")["completed_candidates"]) == 5
    monkeypatch.setattr(staged, "train_models", actual_train)
    changed = config.model_copy(deep=True)
    changed.seeds.model += 1
    with pytest.raises(ValueError, match="Resume configuration"):
        staged.select_three_stage(prepared, changed, protocol, output.parent, resume=output)
    assert staged.select_three_stage(prepared, config, protocol, output.parent, resume=output,
                                     progress=lambda _: None) == output
    assert file_sha256(output / "stages/01_smote_ratio/decision.json") == frozen_hash
    # 12 benchmark fits, 16 SMOTE fits, with completed pre-interruption fit reused.
    assert len(fitted) == 28
    assert sum(length == 400 for _, _, _, length in fitted) == 12
    bundle = load_bundle(output)
    selection = staged.read_json(output / "selection.json")
    assert bundle.config.smote.sampling_ratio == selection["ratio"]
    assert bundle.config.scoring.threshold == selection["threshold"]
    for model in bundle.models.values():
        assert [model.n_estimators, model.max_depth, model.min_samples_leaf] == selection["forest"]
        assert model.n_jobs == 1
    assert bundle.metadata["test_used"] is False
    root = staged.export_validation(output, prepared, config, tmp_path / "reports", progress=lambda _: None)
    meta, scores = research.load_report(root / "final_validation")
    assert meta["official_test"] is False
    assert scores.threshold.eq(selection["threshold"]).all()
    assert set(reads) == {"train", "validation"}
    frozen_manifest = file_sha256(output / "metadata.json")
    staged.select_three_stage(prepared, config, protocol, output.parent, resume=output, progress=lambda _: None)
    staged.export_validation(output, prepared, config, tmp_path / "reports", progress=lambda _: None)
    assert len(fitted) == 28
    assert file_sha256(output / "metadata.json") == frozen_manifest
    # A subsequent identical experiment reuses both models and continuous reports.
    second = staged.select_three_stage(prepared, config, protocol, output.parent, "reused",
                 reuse_roots=(output,), progress=lambda _: None)
    assert len(fitted) == 28
    assert staged.read_json(second / "selection.json")["threshold"] == selection["threshold"]
    with (output / "selection.json").open("a") as handle:
        handle.write(" ")
    with pytest.raises(ValueError, match="fingerprint"):
        load_bundle(output)


def test_reuse_rejects_dependency_parameters_and_fingerprints(experiment, tmp_path):
    prepared, config, protocol = experiment
    requested = staged.candidate_config(config, .1, protocol.reference_forest)
    model = staged.train_models(prepared, requested, tmp_path / "models", progress=lambda _: None)
    prepared_hash = file_sha256(prepared / "metadata.json")
    manifest = staged.read_json(model / "metadata.json")
    for field, value, message in (("prepared_metadata_sha256", "0" * 64, "provenance"),
                                  ("package_versions", {}, "versions"),
                                  ("rf_parameters", manifest["rf_parameters"] | {"min_samples_leaf": 10}, "parameters")):
        staged.atomic_json(model / "metadata.json", manifest | {field: value})
        with pytest.raises(ValueError, match=message):
            staged.inspect_model(model, prepared_hash)
    staged.atomic_json(model / "metadata.json", manifest)
    with (model / "rf.joblib").open("ab") as handle:
        handle.write(b"corruption")
    with pytest.raises(ValueError, match="fingerprint"):
        staged.inspect_model(model, prepared_hash)
