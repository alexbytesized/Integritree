import json

import pytest
import yaml
from pydantic import ValidationError

from integritree.config import EvaluationConfig, ExperimentConfig, UnresolvedConfigurationError, load_experiment, main
from integritree.settings import BACKEND_ROOT


def test_committed_config_ready_for_baseline_training_but_not_evaluation():
    config = load_experiment(BACKEND_ROOT / "configs/experiment.yaml")
    assert config.dataset.row_count == 6362620
    assert config.target == "isFraud"
    assert config.split.model_dump() == {
        "training": 0.8, "validation": 0.1, "testing": 0.1, "stratified": True,
    }
    assert config.random_forest.n_estimators == 100
    assert config.preprocessing.features == [
        "hour_of_day", "day_of_week", "type_CASH_IN", "type_CASH_OUT",
        "type_DEBIT", "type_PAYMENT", "type_TRANSFER", "log_amount",
        "is_zero_amount", "is_merchant_origin", "is_merchant_dest",
    ]
    assert config.seeds.split == 42
    assert config.preprocessing.time_convention == "simulation_step_1_hour_0_day_0"
    assert config.preprocessing.normalization == "minmax"
    assert config.preprocessing.missing_values == "median_unknown"
    assert config.preprocessing.duplicates == "drop_exact"
    config.require_ready("prepare")
    assert config.unresolved_fields("prepare") == []
    config.require_ready("train")
    with pytest.raises(UnresolvedConfigurationError, match="evaluation.pr_auc_method"):
        config.require_ready("evaluate")
    assert config.smote.method == "smote_encoded"
    assert config.smote.sampling_ratio == 1.0
    assert config.scoring.probability_method == "mean_tree_probability"
    assert config.scoring.threshold == 0.5
    assert config.scoring.tie_policy == "fraud"
    assert config.random_forest.tuning_procedure == "fixed_baseline"
    assert config.evaluation.pr_auc_method is None
    assert config.evaluation.percentage_difference == "signed_over_mean"
    assert config.seeds.model == config.seeds.smote == 42
    assert config.smote.k_neighbors == 5


def test_comparison_method_preserves_legacy_snapshots_and_rejects_unknown_methods():
    legacy = EvaluationConfig.model_validate({"percentage_difference": "absolute_over_mean"})
    assert EvaluationConfig.model_validate_json(legacy.model_dump_json()) == legacy
    assert EvaluationConfig().percentage_difference == "absolute_over_mean"
    with pytest.raises(ValidationError, match="percentage_difference"):
        EvaluationConfig.model_validate({"percentage_difference": "relative_to_baseline"})


@pytest.mark.parametrize("updates,field", [
    ({"target": "isFlaggedFraud"}, "target"),
    ({"split": {"training": 0.7}}, "split.training"),
    ({"random_forest": {"n_estimators": 0}}, "random_forest.n_estimators"),
    ({"random_forest": {"n_estimators": True}}, "random_forest.n_estimators"),
    ({"seeds": {"split": -1}}, "seeds.split"),
    ({"smote": {"sampling_ratio": 1.5}}, "smote.sampling_ratio"),
    ({"scoring": {"threshold": float("nan")}}, "scoring.threshold"),
    ({"scoring": {"threshold": True}}, "scoring.threshold"),
    ({"preprocessing": {"features": ["isFraud"]}}, "preprocessing.features"),
    ({"preprocessing": {"features": []}}, "preprocessing.features"),
    ({"preprocessing": {"features": ["log_amount", "log_amount"]}}, "preprocessing"),
    ({"unknown_setting": 1}, "unknown_setting"),
])
def test_invalid_configuration_rejected(draft, updates, field):
    with pytest.raises(ValidationError, match=field):
        ExperimentConfig.model_validate(draft | updates)


def test_prepare_gate_does_not_require_training_parameters(draft):
    draft["preprocessing"] = {
        "features": ["log_amount"], "time_convention": "Synthetic test convention only",
        "normalization": "none", "missing_values": "median_unknown", "duplicates": "drop_exact",
    }
    draft["seeds"] = {"split": 7}
    config = ExperimentConfig.model_validate(draft)
    config.require_ready("prepare")
    with pytest.raises(UnresolvedConfigurationError, match="random_forest.n_estimators"):
        config.require_ready("train")
    assert "shap.output_space" not in config.unresolved_fields("train")
    assert "shap.output_space" in config.unresolved_fields("explain")


@pytest.mark.parametrize("text", [
    "", "[]", "experiment_name: one\nexperiment_name: two\n",
    "experiment_name: [", "!!python/object/apply:os.system ['echo unsafe']",
])
def test_invalid_yaml_fails_without_execution(tmp_path, text):
    path = tmp_path / "experiment.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        load_experiment(path)


def test_missing_file_reports_path(tmp_path):
    with pytest.raises(ValueError, match="missing.yaml"):
        load_experiment(tmp_path / "missing.yaml")


def test_nested_duplicate_yaml_key_rejected(tmp_path, draft):
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(draft) + "seeds:\n  split: 1\n  split: 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        load_experiment(path)


def test_cli_valid_draft_and_gate(settings, monkeypatch, capsys):
    monkeypatch.setenv("INTEGRITREE_BACKEND_ROOT", str(settings.backend_root))
    monkeypatch.setattr("sys.argv", ["integritree-config"])
    assert main() == 0
    assert json.loads(capsys.readouterr().out)["valid"] is True
    monkeypatch.setattr("sys.argv", ["integritree-config", "--require-stage", "train"])
    assert main() == 2
    error = json.loads(capsys.readouterr().out)["error"]
    assert "seeds.split" in error
    assert "smote.k_neighbors" in error


def test_unknown_stage_rejected(draft):
    with pytest.raises(ValueError, match="Unknown stage"):
        ExperimentConfig.model_validate(draft).require_ready("typo")
