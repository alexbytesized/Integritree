import joblib
import numpy as np
import pandas as pd
import pytest
import run as runner
from integritree.config import load_experiment
from integritree.ml.artifacts import ModelBundle
from integritree.ml.preprocessing import FittedPreprocessor
from sklearn.ensemble import RandomForestClassifier


@pytest.mark.parametrize(
    "field,value",
    [
        ("n_estimators", 100),
        ("seed", 1),
        ("forests", [[20, 1]]),
        ("background_rows", 100),
    ],
)
def test_protocol_is_fixed(tmp_path, field, value):
    settings = runner.read_json(runner.ROOT / "config.json")
    settings[field] = value
    path = tmp_path / "config.json"
    runner.atomic_json(path, settings)
    with pytest.raises(ValueError):
        runner.read_config(path)


def test_csv_values_duplicates_and_labels(tmp_path):
    frame = pd.DataFrame(np.full((4, 11), 0.25), columns=runner.FEATURE_COLUMNS)
    frame["isFraud"] = [0, 1, 1, 0]
    path = tmp_path / "data.csv"
    frame[list(reversed(frame.columns))].to_csv(path, index=False)
    X, y, ids = runner.load_csv(path)
    np.testing.assert_array_equal(X, frame[runner.FEATURE_COLUMNS])
    np.testing.assert_array_equal(y, [0, 1, 1, 0])
    np.testing.assert_array_equal(ids, [1, 2, 3, 4])
    frame["isFraud"] = [0.0, 1.0, 0.5, 0.0]
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="binary"):
        runner.load_csv(path)


def test_model_checkpoint_reuse_context_and_corruption(tmp_path, monkeypatch):
    X = np.random.default_rng(42).random((80, 11))
    y = np.array([0, 1] * 40)
    params = {
        "n_estimators": 4,
        "max_depth": 4,
        "min_samples_leaf": 1,
        "random_state": 42,
        "n_jobs": 1,
    }
    checkpoint = runner.fit_one(tmp_path, "rf", X, y, params, "training-context")

    def forbidden(*args, **kwargs):
        raise AssertionError("Completed fit must be reused")

    monkeypatch.setattr(RandomForestClassifier, "fit", forbidden)
    assert (
        runner.fit_one(tmp_path, "rf", X, y, params, "training-context") == checkpoint
    )
    with pytest.raises(ValueError, match="checkpoint"):
        runner.checkpoint_model(tmp_path, "rf", "different-context")
    with (tmp_path / "rf.joblib").open("ab") as handle:
        handle.write(b"corruption")
    with pytest.raises(ValueError, match="checkpoint"):
        runner.checkpoint_model(tmp_path, "rf", "training-context")


def fixture_experiment(tmp_path, monkeypatch):
    actual_root = runner.ROOT
    root = tmp_path / "isolated"
    root.mkdir()
    rng = np.random.default_rng(42)
    raw = pd.DataFrame(
        {
            "step": rng.integers(1, 744, 1240),
            "type": rng.choice(["TRANSFER", "CASH_OUT"], 1240),
            "amount": rng.uniform(1, 10000, 1240),
            "nameOrig": "C_TEST",
            "nameDest": "C_DEST",
        }
    )
    prep = FittedPreprocessor.fit(raw.iloc[:1200])
    features = prep.transform(raw)
    truth = np.zeros(1240, dtype="uint8")
    truth[np.argsort(raw.amount.iloc[:1200])[-8:]] = 1
    truth[1200::8] = 1
    config = load_experiment(
        actual_root.parent / "backend/configs/experiment.yaml"
    ).model_copy(deep=True)
    config.random_forest.n_estimators = 200
    config.random_forest.max_depth = 20
    config.random_forest.min_samples_leaf = 1
    config.smote.sampling_ratio = 0.01
    validation = root / "validation.csv"
    features.iloc[1200:].assign(isFraud=truth[1200:]).to_csv(validation, index=False)
    train_path = root / "training"
    train_path.mkdir()
    features.iloc[:1200].to_parquet(train_path / "train_features.parquet", index=False)
    pd.DataFrame({"actual_label": truth[:1200]}).to_parquet(
        train_path / "split_manifest.parquet", index=False
    )
    prep.save(train_path / "preprocessing.json")
    bundles, plan = {}, []
    for depth, leaf in runner.GRID:
        name = runner.candidate_name(depth, leaf)
        source = None
        if depth == 20:
            source = root / "sources" / name
            source.mkdir(parents=True)
            prep.save(source / "preprocessing.json")
            params = runner.forest_parameters(config, jobs=1) | {
                "min_samples_leaf": leaf
            }
            models = {
                kind: RandomForestClassifier(**params).fit(
                    features.iloc[:1200].to_numpy(), truth[:1200]
                )
                for kind in runner.KINDS
            }
            for kind, model in models.items():
                joblib.dump(model, source / f"{kind}.joblib")
            bundles[str(source)] = ModelBundle(models, prep, config, {})
        plan.append(
            {
                "candidate": name,
                "max_depth": depth,
                "min_samples_leaf": leaf,
                "source": str(source) if source else None,
                "new_fits": 0 if source else 2,
            }
        )
    previous = root / "previous"
    previous.mkdir()
    rows = []
    for kind in runner.KINDS:
        for mode in ("fixed_43", "selected"):
            row = runner.metric_row(
                truth[1200:],
                np.linspace(0, 1, 40),
                0.43,
                config.evaluation,
                "depth_10_leaf_1",
                kind,
                mode,
            )
            rows.append(row | {"mean_time_abs_pp": 0.1, "time_share_abs": 0.2})
    pd.DataFrame(rows).to_csv(previous / "comparison.csv", index=False)
    runner.atomic_json(
        previous / "manifest.json",
        {"validation_csv_sha256": runner.file_sha256(validation)},
    )
    inventory = {str(p): runner.file_sha256(p) for p in root.rglob("*") if p.is_file()}
    settings = runner.read_json(actual_root / "config.json") | {
        "prepared": str(train_path),
        "validation_csv": str(validation),
        "previous_results": str(previous),
        "saved_candidates": str(root / "sources"),
    }
    runner.atomic_json(root / "config.json", settings)
    metadata = {"splits": {"train": {"rows": 1200, "legitimate": 1192}}}
    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(
        runner,
        "inspect_sources",
        lambda _: (config.model_dump(), plan, inventory.copy(), metadata),
    )
    monkeypatch.setattr(runner, "load_bundle", lambda p: bundles[str(p)])
    splits = []

    def load_training(path, split):
        splits.append(split)
        assert split == "train" and path == train_path
        return features.iloc[:1200], pd.Series(truth[:1200]), pd.Series(np.arange(1200))

    monkeypatch.setattr(runner, "load_prepared_split", load_training)

    def forbidden_transform(*args, **kwargs):
        raise AssertionError("Already engineered CSV must not be transformed")

    monkeypatch.setattr(FittedPreprocessor, "transform", forbidden_transform)
    return root, features.iloc[:1200].to_numpy(), truth[:1200], splits


def test_interrupted_fit_resume_shared_smote_and_complete_reports(
    tmp_path, monkeypatch
):
    root, train, labels, splits = fixture_experiment(tmp_path, monkeypatch)
    original_fit, original_smote = runner.fit_one, runner.resample_training
    fits, smotes = [], []

    def interrupted(folder, kind, X, y, params, context):
        if len(fits) == 1:
            raise KeyboardInterrupt("synthetic interruption")
        fits.append((folder.name, kind))
        return original_fit(folder, kind, X, y, params, context)

    def audited_smote(X, y, config):
        np.testing.assert_array_equal(X, train)
        np.testing.assert_array_equal(y, labels)
        smotes.append(len(y))
        return original_smote(X, y, config)

    monkeypatch.setattr(runner, "fit_one", interrupted)
    monkeypatch.setattr(runner, "resample_training", audited_smote)
    with pytest.raises(KeyboardInterrupt):
        runner.run(root / "config.json", "synthetic", execute=True)
    output = root / "outputs/synthetic"
    assert not (output / ".run.lock").exists()
    assert runner.read_json(output / "manifest.json")["status"] == "interrupted"

    def remaining(folder, kind, X, y, params, context):
        fits.append((folder.name, kind))
        return original_fit(folder, kind, X, y, params, context)

    monkeypatch.setattr(runner, "fit_one", remaining)
    runner.run(root / "config.json", execute=True, resume="synthetic")
    assert len(fits) == len(set(fits)) == 6
    assert smotes == [1200]
    assert splits == ["train", "train"]
    manifest = runner.read_json(output / "manifest.json")
    assert manifest["status"] == "complete" and manifest["new_fits_completed"] == 6
    assert len(manifest["completed"]) == 6
    assert runner.read_json(output / "verification.json")["thresholds_verified"] == 600
    runner.validate_files(output, manifest["files"])
    combined = pd.read_csv(output / "comparison.csv")
    assert len(combined) == 24
    assert combined.groupby(["candidate", "model"]).pr_auc.nunique().eq(1).all()
    for (name, mode), pair in combined.groupby(["candidate", "mode"]):
        assert len(pair) == 2, (name, mode)
        np.testing.assert_allclose(pair.pair_mean_f1, pair.f1.mean())
    with pytest.raises(FileExistsError):
        runner.run(root / "config.json", "synthetic", execute=True)
    with pytest.raises(ValueError, match="unfinished"):
        runner.run(root / "config.json", execute=True, resume="synthetic")

    # Simulate interruption after pair checkpoints, before final completion.
    manifest["status"] = "interrupted"
    runner.atomic_json(output / "manifest.json", manifest)

    def forbidden_repeat(*args, **kwargs):
        raise AssertionError("Completed training and evaluation must be reused")

    monkeypatch.setattr(runner, "fit_one", forbidden_repeat)
    monkeypatch.setattr(runner, "shap_values", forbidden_repeat)
    monkeypatch.setattr(runner, "resample_training", forbidden_repeat)
    runner.run(root / "config.json", execute=True, resume="synthetic")
    assert runner.read_json(output / "manifest.json")["status"] == "complete"
    assert splits == ["train", "train"]


def test_preflight_no_training_and_resume_mismatch(tmp_path, monkeypatch):
    root, _, _, splits = fixture_experiment(tmp_path, monkeypatch)

    def forbidden(*args, **kwargs):
        raise AssertionError("Preflight must not train")

    monkeypatch.setattr(runner, "train_missing", forbidden)
    output = runner.run(root / "config.json", "preflight")
    assert runner.read_json(output / "manifest.json")["status"] == "preflight_complete"
    assert not splits
    manifest = runner.read_json(output / "manifest.json")
    manifest.update(status="interrupted", mode="execute", identity_signature="wrong")
    runner.atomic_json(output / "manifest.json", manifest)
    with pytest.raises(ValueError, match="Resume inputs"):
        runner.run(root / "config.json", execute=True, resume="preflight")
