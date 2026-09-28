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
from integritree.ml.selection_utils import exact_mean_f1
from integritree.ml.threshold_search import percent_grid_thresholds
from integritree.ml.training import resample_training, memory_preflight


def test_declared_protocol_and_ties():
    protocol = staged.load_protocol(BACKEND_ROOT / "configs/validation_three_stage.yaml")
    assert protocol.ratios == [.01, .02, .05, .1, .2, 1 / 3, .5, 1]
    assert staged.SelectionProtocol().model_dump() == protocol.model_dump()
    assert protocol.reference_forest == (100, 10, 1)
    assert len(protocol.forests()) == 12
    assert set(protocol.forests()) == {(t, d, l) for t in (100, 200) for d in (10, 20) for l in (1, 10, 50)}
    ratios = [{"ratio": r, "rf_smote": {"pr_auc": .8}} for r in reversed(protocol.ratios)]
    assert staged.choose_ratio(ratios)["ratio"] == .01
    forests = [{"forest": f, "mean_ap": .8} for f in reversed(protocol.forests())]
    assert staged.choose_forest(forests)["forest"] == (100, 10, 50)
    ratios[-1]["rf_smote"]["pr_auc"] = .7
    assert staged.choose_ratio(ratios)["ratio"] == .02
    forests[0]["mean_ap"] = .9
    assert staged.choose_forest(forests) is forests[0]
    for change in ({"ratios": []}, {"ratios": [float("nan")]}, {"ratios": [0]},
                   {"leaves": [1, 1]}, {"selection_split": "test"}, {"trees": [0]},
                   {"threshold_candidates": "unsupported_policy"}):
        with pytest.raises(ValueError):
            staged.SelectionProtocol.model_validate(protocol.model_dump() | change)


@pytest.mark.parametrize("seed", range(10))
def test_percent_grid_matches_exhaustive(seed):
    rng = np.random.default_rng(seed)
    labels = np.r_[0, 1, rng.integers(0, 2, size=28)]
    scores = rng.choice([0, .125, .25, .5, .75, .96, .9876543210987654, 1.], size=(2, 30))
    best, curve = percent_grid_thresholds(labels, *scores)
    candidates = [k / 100 for k in range(1, 101)]
    expected = max(candidates, key=lambda t: (exact_mean_f1(labels, *scores, t),
                                             -abs(round(t * 100) - 50), t))
    assert best["threshold"] == expected
    assert Fraction(best["mean_f1_fraction"]) == exact_mean_f1(labels, *scores, expected)
    assert best["candidate_count"] == len(candidates)
    assert list(curve.threshold) == candidates
    assert list(curve.threshold_percent) == list(range(1, 101))
    assert best["candidate_count"] == 100 and 0 not in candidates
    for row in curve.to_dict("records"):
        for name, values in zip(("rf", "rf_smote"), scores):
            predicted = values >= row["threshold"]
            for field, actual, prediction in (("tp", 1, True), ("fp", 0, True),
                                               ("tn", 0, False), ("fn", 1, False)):
                assert row[f"{name}_{field}"] == np.count_nonzero((labels == actual) & (predicted == prediction))
    repeated, same = percent_grid_thresholds(labels, *scores)
    assert repeated == best
    pd.testing.assert_frame_equal(curve, same, check_exact=True)


def test_high_threshold_and_closest_half_ties():
    scores = [.96, .97, .9876543210987654, .99]
    best, _ = percent_grid_thresholds([0, 0, 1, 1], scores, scores)
    assert best["threshold"] == .98
    assert percent_grid_thresholds([0, 1], [0, 1], [0, 1])[0]["threshold"] == .5
    # Equidistant decimal cutoffs .49 and .51 must prefer .51.
    best, _ = percent_grid_thresholds([1, 0, 0, 1], [.49, .50, .50, .51], [.49, .50, .50, .51])
    assert best["threshold"] == .51
    with pytest.raises(ValueError, match="both"):
        percent_grid_thresholds([0, 0], [0, 1], [0, 1])


@pytest.mark.parametrize("scores, expected", [([0., .01], .01), ([.99, 1.], 1.)])
def test_percent_grid_endpoints_and_equality(scores, expected):
    best, curve = percent_grid_thresholds([0, 1], scores, scores)
    assert best["threshold"] == expected
    row = curve.loc[curve.threshold == expected].iloc[0]
    assert row.rf_tp == 1 and row.rf_fp == 0


@pytest.mark.parametrize("ratio", [.01, .02, .05, .1, .2, 1 / 3, .5, 1.])
def test_all_resampling_counts_and_memory(ratio):
    config = load_experiment(BACKEND_ROOT / "configs/experiment.yaml")
    config.smote.sampling_ratio = ratio
    values = np.random.default_rng(42).random((2011, len(FEATURE_COLUMNS)))
    labels = np.r_[np.ones(10, dtype="uint8"), np.zeros(2001, dtype="uint8")]
    original = values.copy()
    resampled, targets, audit = resample_training(values, labels, config)
    minority = int(2001 * ratio)
    assert len(targets) == 2001 + minority
    assert audit["class_counts"] == {"0": 2001, "1": minority}
    assert audit["generated_fraud_rows"] == minority - 10
    assert audit["requested_sampling_ratio"] == ratio
    assert audit["achieved_sampling_ratio"] == minority / 2001
    np.testing.assert_array_equal(resampled[:2011], original)
    np.testing.assert_array_equal(targets[:2011], labels)
    assert (targets[2011:] == 1).all()
    preflight = memory_preflight(2011, 2001, ratio)
    assert preflight["estimated_resampled_rows"] == len(targets)
    assert preflight["estimated_array_bytes"] == (3 * 2011 + 4 * len(targets)) * len(FEATURE_COLUMNS) * 8 + 64 * 1024**2


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
    protocol = staged.SelectionProtocol(ratios=[.1, .2, 1 / 3, .5, 1.], trees=[3, 4], depths=[2, 3], leaves=[1, 2, 4], reference_forest=(3, 2, 1))
    return prepared, config, protocol


@pytest.fixture
def sparse_experiment(tmp_path, raw_record):
    """Enough minority neighbors, with original prevalence below 1:100."""
    frame = pd.DataFrame([raw_record | {"step": i + 1, "amount": float(i + 1),
             "nameOrig": f"C_{i}", "type": "TRANSFER" if i % 2 else "CASH_OUT",
             "isFraud": int(i % 200 == 0)} for i in range(4000)]).loc[:, RAW_COLUMNS]
    source = tmp_path / "sparse.csv"
    frame.to_csv(source, index=False)
    config = load_experiment(BACKEND_ROOT / "configs/experiment.yaml").model_copy(deep=True)
    config.dataset = config.dataset.model_copy(update={"filename": source.name,
                  "sha256": file_sha256(source), "row_count": len(frame)})
    prepared = prepare_dataset(source, config, tmp_path / "prepared", "sparse", progress=lambda _: None)
    protocol = staged.SelectionProtocol(trees=[3], depths=[2], leaves=[1], reference_forest=(3, 2, 1))
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


def test_stage_one_only_stops_and_can_resume_without_refitting(sparse_experiment, tmp_path, monkeypatch):
    prepared, config, protocol = sparse_experiment
    import integritree.ml.training as training
    import integritree.ml.research as research
    training_loader, evaluation_loader = training.load_prepared_split, research.load_prepared_split
    def train_only(path, split):
        assert split == "train"
        return training_loader(path, split)
    def validation_only(path, split):
        assert split == "validation"
        return evaluation_loader(path, split)
    monkeypatch.setattr(training, "load_prepared_split", train_only)
    monkeypatch.setattr(research, "load_prepared_split", validation_only)
    actual_train = staged.train_models
    fitted = []
    def reference_only(path, requested, *args, **kwargs):
        assert (requested.random_forest.n_estimators, requested.random_forest.max_depth,
                requested.random_forest.min_samples_leaf) == protocol.reference_forest
        fitted.append(requested.smote.sampling_ratio)
        return actual_train(path, requested, *args, **kwargs)
    monkeypatch.setattr(staged, "train_models", reference_only)
    def forbidden(*args, **kwargs):
        raise AssertionError("Later-stage selection must not run")
    monkeypatch.setattr(staged, "choose_forest", forbidden)
    monkeypatch.setattr(staged, "percent_grid_thresholds", forbidden)
    output = staged.select_three_stage(prepared, config, protocol, tmp_path / "selection",
                         "stage_one", stop_after_stage=1, progress=lambda _: None)
    assert fitted == protocol.ratios
    assert len(staged.read_json(output / "search_progress.json")["completed_candidates"]) == 8
    manifest = staged.read_json(output / "metadata.json")
    assert manifest["status"] == "awaiting_next_stage"
    assert manifest["completed_stage"] == 1
    assert manifest["stage"] == "ratio_selected"
    assert manifest["execution_history"][-1]["stop_after_stage"] == 1
    assert not (output / "selection.json").exists()
    assert not (output / "stages/02_random_forest").exists()
    assert not (output / "stages/03_threshold").exists()
    with pytest.raises(ValueError, match="incomplete"):
        load_bundle(output)
    plan_hash = file_sha256(output / "search_plan.json")
    decision_hash = file_sha256(output / "stages/01_smote_ratio/decision.json")
    staged.select_three_stage(prepared, config, protocol, output.parent, resume=output,
                              stop_after_stage=1, progress=lambda _: None)
    assert fitted == protocol.ratios
    assert file_sha256(output / "search_plan.json") == plan_hash
    assert file_sha256(output / "stages/01_smote_ratio/decision.json") == decision_hash
    report = staged.export_ratio_stage(output, tmp_path / "reports")
    assert staged.read_json(report / "metadata.json")["test_used"] is False
    assert file_sha256(report / "decision.json") == decision_hash
    assert len(staged.read_json(report / "decision.json")["candidates"]) == 8
    changed = protocol.model_copy(update={"ratios": [.1, .2, 1 / 3, .5, 1.]})
    with pytest.raises(ValueError, match="Resume configuration"):
        staged.select_three_stage(prepared, config, changed, output.parent, resume=output,
                                  stop_after_stage=1, progress=lambda _: None)
    assert file_sha256(output / "search_plan.json") == plan_hash
    with (output / "stages/01_smote_ratio/decision.json").open("a") as handle:
        handle.write(" ")
    with pytest.raises(ValueError, match="fingerprint"):
        staged.select_three_stage(prepared, config, protocol, output.parent, resume=output,
                                  stop_after_stage=1, progress=lambda _: None)


def test_obsolete_selection_bundle_rejected(tmp_path):
    staged.atomic_json(tmp_path / "metadata.json", {
        "kind": "validation_selection", "schema_version": 1, "status": "complete"})
    with pytest.raises(ValueError, match="three-stage schema-2"):
        load_bundle(tmp_path)


def test_protocol_rejects_removed_amendment_fields():
    with pytest.raises(ValueError, match="Extra inputs"):
        staged.SelectionProtocol(revision=2, amendment_note="obsolete")


def test_fresh_stage_one_reuses_only_its_benchmark_and_releases_reports(sparse_experiment, tmp_path, monkeypatch):
    import weakref
    prepared, config, protocol = sparse_experiment
    real_load, real_train = staged.load_report, staged.train_models
    frames, sources = [], []
    output = tmp_path / "runs" / "fresh"
    def tracked_load(path):
        meta, frame = real_load(path)
        frames.append(weakref.ref(frame))
        return meta, frame
    def checked_train(*args, **kwargs):
        assert all(ref() is None for ref in frames)
        source = kwargs.get("reuse_rf_from")
        if sources:
            assert source is not None and source.is_relative_to(output)
        else:
            assert source is None
        sources.append(source)
        return real_train(*args, **kwargs)
    monkeypatch.setattr(staged, "load_report", tracked_load)
    monkeypatch.setattr(staged, "train_models", checked_train)
    staged.select_three_stage(prepared, config, protocol, output.parent, output.name,
                              stop_after_stage=1, progress=lambda _: None)
    plan = staged.read_json(output / "search_plan.json")
    assert plan["reuse"]["models"] == [] and plan["reuse"]["reports"] == []
    assert len(sources) == 8


def test_completed_three_stage_selection_allows_synthetic_test(experiment, tmp_path):
    from integritree.ml.research import evaluate_run, load_report
    prepared, config, _ = experiment
    protocol = staged.SelectionProtocol(ratios=[.1, .2, 1 / 3, .5, 1.], trees=[3], depths=[2], leaves=[1], reference_forest=(3, 2, 1))
    output = staged.select_three_stage(prepared, config, protocol, tmp_path / "runs",
                                      progress=lambda _: None)
    bundle = load_bundle(output)
    report = evaluate_run(output, prepared, config, tmp_path / "reports", "test", progress=lambda _: None)
    meta, scores = load_report(report)
    assert meta["official_test"] is True
    assert scores.threshold.eq(bundle.config.scoring.threshold).all()
    effective = staged.read_json(report / "evaluation_configuration.json")
    assert effective["effective_random_forest"]["tuning_procedure"] == "three_stage_validation_selected"


@pytest.fixture
def control_experiment(experiment):
    prepared, config, _ = experiment
    protocol = staged.SelectionProtocol(ratios=[.1, .2], trees=[3, 4], depths=[2],
                                        leaves=[1, 2], reference_forest=(3, 2, 1))
    return prepared, config, protocol


def test_stage_two_pause_resume_matches_uninterrupted(control_experiment, tmp_path, monkeypatch):
    import integritree.ml.training as training
    import integritree.ml.research as research
    prepared, config, protocol = control_experiment
    train_loader, eval_loader = training.load_prepared_split, research.load_prepared_split
    def train_only(path, split):
        assert split == "train"
        return train_loader(path, split)
    def validation_only(path, split):
        assert split == "validation"
        return eval_loader(path, split)
    monkeypatch.setattr(training, "load_prepared_split", train_only)
    monkeypatch.setattr(research, "load_prepared_split", validation_only)
    def forbidden(*args, **kwargs):
        raise AssertionError("Stage 2 must not enter threshold selection")
    monkeypatch.setattr(staged, "percent_grid_thresholds", forbidden)
    output = staged.select_three_stage(prepared, config, protocol, tmp_path / "runs", "paused_run",
                                      stop_after_stage=1, progress=lambda _: None)
    preserved = {p: file_sha256(p) for p in [output / "search_plan.json",
                 *list((output / "stages/01_smote_ratio").glob("*"))]}
    initial = staged.read_json(output / "search_progress.json")["completed_candidates"]
    real_train = staged.train_models
    calls = []
    def pause_during_training(*args, **kwargs):
        calls.append(1)
        assert len(calls) == 1, "Started another fit after a pause request"
        first = staged.request_pause(output)
        before = (output / "pause_request.json").read_bytes()
        assert staged.request_pause(output) == first
        assert (output / "pause_request.json").read_bytes() == before
        assert first["status"] == "pause_requested"
        assert staged.read_json(output / "metadata.json")["status"] == "running"
        return real_train(*args, **kwargs)
    monkeypatch.setattr(staged, "train_models", pause_during_training)
    staged.select_three_stage(prepared, config, protocol, output.parent, resume=output,
                              stop_after_stage=2, progress=lambda _: None)
    meta = staged.read_json(output / "metadata.json")
    completed = staged.read_json(output / "search_progress.json")["completed_candidates"]
    assert meta["status"] == "paused" and meta["completed_stage"] == 1
    assert meta["active_stage"] == 2 and meta["active_candidate"] is None
    assert len(completed) == len(initial) + 1
    row = completed[meta["last_completed_candidate"]]
    assert staged.read_json(staged.resolve_reference(row["model"], output) / "metadata.json")["status"] == "complete"
    assert staged.read_json(staged.resolve_reference(row["report"], output) / "metadata.json")["status"] == "complete"
    assert not (output / "pause_request.json").exists()
    assert not (output / "stages/02_random_forest/decision.json").exists()
    with staged.run_lock(output):
        pass  # Paused worker released its OS lock.
    with pytest.raises(ValueError, match="incomplete"):
        staged.export_forest_stage(output, tmp_path / "reports")
    with pytest.raises(ValueError, match="incomplete"):
        load_bundle(output)
    with pytest.raises(ValueError, match="active"):
        staged.request_pause(output)
    already = {(r["ratio"], tuple(r["forest"])) for r in completed.values()}
    resumed_fits = []
    def only_remaining(path, requested, *args, **kwargs):
        key = (requested.smote.sampling_ratio, (requested.random_forest.n_estimators,
               requested.random_forest.max_depth, requested.random_forest.min_samples_leaf))
        assert key not in already
        resumed_fits.append(key)
        return real_train(path, requested, *args, **kwargs)
    monkeypatch.setattr(staged, "train_models", only_remaining)
    staged.select_three_stage(prepared, config, protocol, output.parent, resume=output,
                              stop_after_stage=2, progress=lambda _: None)
    assert len(resumed_fits) == 2
    assert all(file_sha256(p) == digest for p, digest in preserved.items())
    meta = staged.read_json(output / "metadata.json")
    assert (meta["status"], meta["stage"], meta["completed_stage"]) == ("awaiting_next_stage", "forest_selected", 2)
    assert not (output / "selection.json").exists()
    assert not (output / "stages/03_threshold").exists()
    with pytest.raises(ValueError, match="incomplete"):
        load_bundle(output)
    report = staged.export_forest_stage(output, tmp_path / "reports")
    review = staged.read_json(report / "metadata.json")
    assert review["test_used"] is False and review["status"] == "complete"
    for name, digest in review["files"].items():
        assert file_sha256(report / name) == digest
    monkeypatch.setattr(staged, "train_models", real_train)
    uninterrupted = staged.select_three_stage(prepared, config, protocol, output.parent, "uninterrupted",
                                             stop_after_stage=2, progress=lambda _: None)
    actual = staged.read_json(output / "stages/02_random_forest/decision.json")
    expected = staged.read_json(uninterrupted / "stages/02_random_forest/decision.json")
    assert actual["chosen_key"] == expected["chosen_key"]
    for left, right in zip(actual["candidates"], expected["candidates"]):
        assert {k:v for k,v in left.items() if k not in ("model", "report")} == {
                k:v for k,v in right.items() if k not in ("model", "report")}
        _, lpred = research.load_report(staged.resolve_reference(left["report"], output))
        _, rpred = research.load_report(staged.resolve_reference(right["report"], uninterrupted))
        pd.testing.assert_frame_equal(lpred[["rf_risk_score", "rf_smote_risk_score"]],
                                      rpred[["rf_risk_score", "rf_smote_risk_score"]], check_exact=True)
    with (output / "stages/02_random_forest/candidates.csv").open("a") as handle:
        handle.write("corrupt")
    with pytest.raises(ValueError, match="fingerprint"):
        staged.export_forest_stage(output, tmp_path / "reports")
    with pytest.raises(ValueError, match="fingerprint"):
        staged.select_three_stage(prepared, config, protocol, output.parent, resume=output, stop_after_stage=2)


def test_pause_during_last_candidate_freezes_stage_two(control_experiment, tmp_path, monkeypatch):
    prepared, config, protocol = control_experiment
    output = tmp_path / "runs" / "last"
    real_train = staged.train_models
    requested_pause = []
    def last_candidate(path, requested, *args, **kwargs):
        forest = (requested.random_forest.n_estimators, requested.random_forest.max_depth,
                  requested.random_forest.min_samples_leaf)
        if forest == protocol.forests()[-1]:
            requested_pause.append(staged.request_pause(output))
        return real_train(path, requested, *args, **kwargs)
    monkeypatch.setattr(staged, "train_models", last_candidate)
    staged.select_three_stage(prepared, config, protocol, output.parent, output.name,
                              stop_after_stage=2, progress=lambda _: None)
    assert len(requested_pause) == 1
    meta = staged.read_json(output / "metadata.json")
    assert meta["status"] == "awaiting_next_stage" and meta["completed_stage"] == 2
    assert not (output / "pause_request.json").exists()
    assert not (output / "stages/03_threshold").exists()
    with pytest.raises(ValueError, match="active"):
        staged.request_pause(output)


@pytest.mark.parametrize("status", ["paused", "complete", "failed", "awaiting_next_stage", "running"])
def test_pause_rejects_inactive_or_stale_worker(tmp_path, status):
    staged.atomic_json(tmp_path / "metadata.json", {"kind": "validation_selection", "schema_version": 2,
                       "status": status, "active_stage": 2, "worker_pid": 999999,
                       "execution_history": [{"started_at": "synthetic"}]})
    with pytest.raises(ValueError, match="active"):
        staged.request_pause(tmp_path)
    assert not (tmp_path / "pause_request.json").exists()


def test_paused_cli_does_not_export_final_results(tmp_path, settings, monkeypatch, capsys):
    output = tmp_path / "paused"
    output.mkdir()
    staged.atomic_json(output / "metadata.json", {"status": "paused"})
    monkeypatch.setattr(staged, "load_settings", lambda: settings)
    monkeypatch.setattr(staged, "load_experiment", lambda _: None)
    monkeypatch.setattr(staged, "load_protocol", lambda _: None)
    monkeypatch.setattr(staged, "select_three_stage", lambda *a, **k: output)
    def forbidden(*args, **kwargs):
        raise AssertionError("A paused run must not export a final report")
    for name in ("export_forest_stage", "export_ratio_stage", "export_validation"):
        monkeypatch.setattr(staged, name, forbidden)
    monkeypatch.setattr("sys.argv", ["select", "--prepared", "synthetic", "--stop-after-stage", "2"])
    assert staged.main() == 0
    response = json.loads(capsys.readouterr().out)
    assert response["status"] == "paused" and response["report"] is None


def test_pause_script_requests_without_reporting_paused(tmp_path):
    import subprocess
    import sys
    staged.atomic_json(tmp_path / "metadata.json", {"kind": "validation_selection", "schema_version": 2,
                       "status": "running", "active_stage": 2, "worker_pid": 123,
                       "execution_history": [{"started_at": "synthetic"}]})
    with staged.run_lock(tmp_path):
        result = subprocess.run([sys.executable, str(BACKEND_ROOT / "scripts/pause_selection.py"),
                                 "--run", str(tmp_path)], capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout)["status"] == "pause_requested"
        assert staged.read_json(tmp_path / "metadata.json")["status"] == "running"
    stale = subprocess.run([sys.executable, str(BACKEND_ROOT / "scripts/pause_selection.py"),
                            "--run", str(tmp_path)], capture_output=True, text=True, check=False)
    assert stale.returncode == 2
    assert "inactive" in json.loads(stale.stderr)["error"]


@pytest.mark.parametrize("interrupt", [False, True])
def test_stage3_reset_preserves_evidence_and_recovers(control_experiment, tmp_path, monkeypatch, interrupt):
    from integritree.ml import stage3_reset as reset
    import integritree.ml.research as research
    prepared, config, protocol = control_experiment
    run = staged.select_three_stage(prepared, config, protocol, tmp_path / "artifacts", "reset_run",
                                     progress=lambda _: None)
    staged.export_ratio_stage(run, tmp_path / "reports")
    staged.export_forest_stage(run, tmp_path / "reports")
    review = staged.export_validation(run, prepared, config, tmp_path / "reports", progress=lambda _: None)
    preserved = {p: file_sha256(p) for folder in ("candidates", "validation", "stages/01_smote_ratio", "stages/02_random_forest")
                 for p in (run / folder).rglob("*") if p.is_file()}
    preserved[run / "search_progress.json"] = file_sha256(run / "search_progress.json")
    # Simulate a completed selection whose declared threshold policy is no longer supported.
    plan = staged.read_json(run / "search_plan.json")
    plan["protocol"]["threshold_candidates"] = "retired_policy"
    staged.atomic_json(run / "search_plan.json", plan)
    meta = staged.read_json(run / "metadata.json")
    meta["search_plan_sha256"] = meta["files"]["search_plan.json"] = file_sha256(run / "search_plan.json")
    staged.atomic_json(run / "metadata.json", meta)
    with pytest.raises(ValueError):
        load_bundle(run)
    with staged.run_lock(run):
        with pytest.raises(staged.RunLockedError):
            reset.reset_stage3(run, tmp_path, protocol)
    with pytest.raises(ValueError, match="Only the threshold"):
        reset.reset_stage3(run, tmp_path, protocol.model_copy(update={"leaves": [1]}))
    def forbidden(*args, **kwargs):
        raise AssertionError("Reset and resume must not fit models")
    monkeypatch.setattr(staged, "train_models", forbidden)
    real_loader = research.load_prepared_split
    def validation_only(path, split):
        assert split == "validation"
        return real_loader(path, split)
    monkeypatch.setattr(research, "load_prepared_split", validation_only)
    actual_delete = reset._delete_file
    calls = []
    if interrupt:
        def interrupted(*args):
            calls.append(1)
            if len(calls) == 2:
                raise OSError("synthetic cleanup interruption")
            return actual_delete(*args)
        monkeypatch.setattr(reset, "_delete_file", interrupted)
        with pytest.raises(OSError, match="synthetic"):
            reset.reset_stage3(run, tmp_path, protocol)
        with pytest.raises(ValueError, match="reset is incomplete"):
            staged.select_three_stage(prepared, config, protocol, run.parent, resume=run)
        monkeypatch.setattr(reset, "_delete_file", actual_delete)
    journal = reset.reset_stage3(run, tmp_path, protocol)
    assert staged.read_json(journal)["status"] == "complete"
    assert staged.read_json(run / "metadata.json")["completed_stage"] == 2
    assert not (run / "selection.json").exists()
    assert not (run / "stages/03_threshold").exists()
    assert not (review / "final_validation").exists()
    for stage in ("stage1_review", "stage2_review"):
        folder = review / stage
        snapshot = staged.read_json(folder / "search_plan.json")
        assert snapshot["protocol"]["threshold_candidates"] == "percent_grid_1_to_100"
        assert "regenerated" in (folder / "SUMMARY.md").read_text()
        for name, digest in staged.read_json(folder / "metadata.json")["files"].items():
            assert file_sha256(folder / name) == digest
    assert all(file_sha256(p) == digest for p, digest in preserved.items())
    staged.select_three_stage(prepared, config, protocol, run.parent, resume=run, progress=lambda _: None)
    staged.export_validation(run, prepared, config, tmp_path / "reports", progress=lambda _: None)
    selected = staged.read_json(run / "selection.json")
    assert selected["candidate_count"] == 100
    assert selected["threshold"] in [k / 100 for k in range(1, 101)]
    assert load_bundle(run).config.scoring.threshold == selected["threshold"]
    assert all(file_sha256(p) == digest for p, digest in preserved.items())
    # Replaying an already completed reset must not delete the new result.
    result_hash = file_sha256(run / "selection.json")
    reset.reset_stage3(run, tmp_path, protocol)
    assert file_sha256(run / "selection.json") == result_hash


def test_stage3_reset_cleanup_boundaries_and_changed_files(tmp_path):
    from integritree.ml import stage3_reset as reset
    inside = tmp_path / "workspace"
    inside.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("keep")
    with pytest.raises(ValueError, match="escapes"):
        reset._delete_file(inside, "../outside.txt", file_sha256(outside))
    with pytest.raises(ValueError, match="escapes"):
        reset.contained(inside, inside)
    target = inside / "output.txt"
    target.write_text("old")
    expected = file_sha256(target)
    target.write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        reset._delete_file(inside, "output.txt", expected)
    assert outside.read_text() == "keep" and target.read_text() == "changed"
