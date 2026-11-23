"""CSV input boundary and an end-to-end saved-model evaluation with fitting blocked."""

import evaluate_csv as runner
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from test_experiment import synthetic_inputs


def sample_frame():
    frame = pd.DataFrame(np.full((3, 11), 0.25), columns=runner.FEATURE_COLUMNS)
    frame["isFraud"] = [0, 1, 1]
    return frame


def test_csv_preserves_values_duplicates_and_separates_labels(tmp_path):
    frame = sample_frame()
    path = tmp_path / "input.csv"
    frame[list(reversed(frame.columns))].to_csv(path, index=False)
    values, labels, ids = runner.load_csv(path)
    np.testing.assert_array_equal(values, frame[runner.FEATURE_COLUMNS])
    np.testing.assert_array_equal(labels, [0, 1, 1])
    np.testing.assert_array_equal(ids, [1, 2, 3])
    assert values.shape == (3, 11)  # Identical feature rows must not be dropped.


@pytest.mark.parametrize(
    "invalid", ["missing", "nan", "infinite", "label", "one_class", "duplicate_header"]
)
def test_invalid_csv_fails_before_evaluation(tmp_path, invalid):
    frame = sample_frame()
    if invalid == "missing":
        frame = frame.drop(columns="day_of_week")
    elif invalid in ("nan", "infinite"):
        frame.iloc[0, 0] = np.nan if invalid == "nan" else np.inf
    elif invalid == "label":
        frame.loc[0, "isFraud"] = 0.5
    elif invalid == "one_class":
        frame["isFraud"] = 1
    else:
        frame.columns = ["hour_of_day"] * len(frame.columns)
    path = tmp_path / "bad.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        runner.load_csv(path)


def test_saved_model_run_without_fit_or_transform(tmp_path, monkeypatch):
    features, labels, bundle = synthetic_inputs()
    source, previous = tmp_path / "original", tmp_path / "previous_cutoffs"
    source.mkdir()
    previous.mkdir()
    supplied = features.iloc[1200:1220].copy()
    truth = labels[1200:1220]
    csv_path = tmp_path / "validation.csv"
    supplied.assign(isFraud=truth).to_csv(csv_path, index=False)
    settings = {
        "seed": 42,
        "shap_uniform_rows": 200,
        "shap_fraud_rows": 100,
        "shap_baseline_flagged_rows": 50,
        "sensitivity_rows": 50,
    }
    features.iloc[:200].to_parquet(source / "shap_background.parquet", index=False)
    bundle.preprocessor.save(source / "preprocessing.json")
    plan, prior_rows, shap_rows = [], [], []
    for depth, leaf in ((10, 1), (8, 1), (6, 1), (10, 10), (10, 50), (8, 10)):
        name = f"depth_{depth}_leaf_{leaf}"
        plan.append(
            {
                "candidate": name,
                "max_depth": depth,
                "min_samples_leaf": leaf,
                "source": None,
            }
        )
        folder = source / name
        folder.mkdir()
        for kind in runner.KINDS:
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=depth,
                min_samples_leaf=leaf,
                random_state=42,
            ).fit(features.iloc[:1200].to_numpy(), labels[:1200])
            joblib.dump(model, folder / f"{kind}.joblib")
            scores = runner.predict(model, supplied.to_numpy())
            prior_rows.append(
                runner.metric_row(
                    truth,
                    scores,
                    0.43,
                    bundle.config.evaluation,
                    name,
                    kind,
                    "fixed_43",
                )
            )
            for cohort in (
                "uniform_validation",
                "actual_fraud",
                "baseline_flagged_rf",
                "baseline_flagged_rf_smote",
            ):
                shap_rows.append(
                    {"candidate": name, "model": kind, "cohort": cohort}
                    | runner.time_summary(np.zeros((2, 11)))
                )
    runner.write_json(
        source / "plan.json",
        {
            "candidates": plan,
            "fixed_configuration": {
                "evaluation": bundle.config.evaluation.model_dump()
            },
        },
    )
    pd.DataFrame(prior_rows).to_csv(source / "metrics.csv", index=False)
    pd.DataFrame(prior_rows).assign(mode="selected").to_csv(
        previous / "metrics_selected.csv", index=False
    )
    pd.DataFrame(shap_rows).to_csv(source / "time_shap.csv", index=False)
    original = {
        "kind": "isolated_forest_regularization",
        "status": "complete",
        "test_used": False,
        "source_preservation_verified": True,
        "plan": plan,
        "settings": settings,
        "files": {
            p.relative_to(source).as_posix(): runner.file_sha256(p)
            for p in source.rglob("*")
            if p.is_file()
        },
    }
    runner.write_json(source / "manifest.json", original)
    runner.write_json(
        previous / "manifest.json",
        {
            "status": "complete",
            "source": str(source),
            "files": {
                "metrics_selected.csv": runner.file_sha256(
                    previous / "metrics_selected.csv"
                )
            },
        },
    )

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Evaluation must not fit models or transform the engineered CSV"
        )

    monkeypatch.setattr(RandomForestClassifier, "fit", forbidden)
    monkeypatch.setattr(runner.FittedPreprocessor, "transform", forbidden)
    output = runner.run(
        csv_path, source, previous, "synthetic_csv", tmp_path / "outputs"
    )
    manifest = runner.read_json(output / "manifest.json")
    assert manifest["status"] == "complete" and manifest["new_fits"] == 0
    assert len(manifest["completed"]) == 6
    assert runner.read_json(output / "verification.json")["thresholds_verified"] == 600
    for path, digest in manifest["files"].items():
        assert runner.file_sha256(output / path) == digest
    result = pd.read_csv(output / "comparison.csv")
    assert len(result) == 24
    for _, pair in result.groupby(["candidate", "model"]):
        assert pair.pr_auc.nunique() == 1
        assert pair.mean_time_abs_pp.nunique() == 1
        assert pair.tp.add(pair.fn).eq(int(truth.sum())).all()
    assert len(pd.read_csv(output / "changes_vs_previous.csv")) == 24
    with pytest.raises(FileExistsError):
        runner.run(csv_path, source, previous, "synthetic_csv", tmp_path / "outputs")
