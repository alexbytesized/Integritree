"""Independent hand-calculated metric/statistical edge cases."""

import numpy as np
import pytest
from integritree.config import load_experiment
from integritree.settings import BACKEND_ROOT
from integritree.ml.evaluation import metrics, comparison, mcnemar


@pytest.fixture
def policy():
    return load_experiment(BACKEND_ROOT / "configs/experiment.yaml").evaluation


def test_hand_calculated_metrics_and_average_precision(policy):
    # TP=1, TN=1, FP=1, FN=1. Ranked labels 1,0,1,0 give AP=(1+2/3)/2.
    result = metrics([1, 0, 1, 0], [0.9, 0.8, 0.3, 0.1], 0.5, policy)
    values = result["metrics"]
    assert values["precision"]["value"] == 0.5
    assert values["recall"]["value"] == 0.5
    assert values["f1"]["value"] == 0.5
    assert values["mcc"]["value"] == 0
    assert values["accuracy"]["value"] == 0.5
    assert values["pr_auc"]["value"] == pytest.approx(5 / 6)
    assert result["confusion_matrix"]["matrix"] == [[1, 1], [1, 1]]


def test_no_predicted_fraud_has_valid_zero_f1_but_undefined_precision(policy):
    result = metrics([0, 1], [0.1, 0.1], 0.5, policy)["metrics"]
    assert result["precision"]["value"] is None
    assert result["recall"]["value"] == 0
    assert result["f1"]["value"] == 0
    assert result["mcc"]["value"] is None


def test_absent_positive_and_perfect_negative_cases(policy):
    result = metrics([0, 0], [0.1, 0.1], 0.5, policy)["metrics"]
    for name in ("precision", "recall", "f1", "mcc", "pr_auc"):
        assert result[name]["value"] is None
    assert result["accuracy"]["value"] == 1
    all_positive = metrics([1, 1], [0.8, 0.9], 0.5, policy)["metrics"]
    assert all_positive["pr_auc"]["value"] == 1
    assert all_positive["mcc"]["value"] is None


def test_perfect_and_reversed_predictions(policy):
    perfect = metrics([0, 1], [0, 1], 0.5, policy)["metrics"]
    assert all(perfect[name]["value"] == 1 for name in perfect)
    inverse = metrics([0, 1], [1, 0], 0.5, policy)["metrics"]
    assert inverse["mcc"]["value"] == -1
    assert inverse["f1"]["value"] == 0


def test_signed_comparison_and_legacy_are_distinct():
    assert comparison(0.2, 0.4, "recall", "signed_over_mean")["value"] == pytest.approx(
        200 / 3
    )
    assert comparison(0.4, 0.2, "recall", "signed_over_mean")["value"] == pytest.approx(
        -200 / 3
    )
    assert comparison(0.4, 0.2, "recall", "absolute_over_mean")[
        "value"
    ] == pytest.approx(200 / 3)
    negative = comparison(-0.5, 0.2, "mcc", "signed_over_mean")
    assert negative["value"] == pytest.approx(0.7)
    assert negative["units"] == "coefficient"
    assert comparison(0, 0, "f1", "signed_over_mean")["value"] is None
    assert (
        comparison(None, 0.3, "mcc", "signed_over_mean")["reason"] == "undefined_input"
    )


def test_mcnemar_primary_and_exact_supplement(policy):
    result = mcnemar([0] * 10, [0] * 8 + [1] * 2, [1] * 8 + [0] * 2, policy)
    assert result["statistic"] == 2.5
    assert result["p_value"] == pytest.approx(0.11384629800665805)
    assert result["supplementary"]["p_value"] == pytest.approx(0.109375)
    assert result["lower_observed_error_model"] == "rf"
    reverse = mcnemar([0] * 10, [1] * 8 + [0] * 2, [0] * 8 + [1] * 2, policy)
    assert reverse["p_value"] == result["p_value"]
    assert reverse["lower_observed_error_model"] == "rf_smote"


def test_mcnemar_no_discordance_and_cutoff(policy):
    result = mcnemar([0, 1], [0, 0], [0, 0], policy)
    assert result["p_value"] == 1 and result["statistic"] is None
    assert result["status"] == "No discordant pairs"
    assert mcnemar([0] * 24, [0] * 24, [1] * 24, policy)["supplementary"] is not None
    assert mcnemar([0] * 25, [0] * 25, [1] * 25, policy)["supplementary"] is None


@pytest.mark.parametrize(
    "labels,scores",
    [
        ([], []),
        ([0, 2], [0.1, 0.9]),
        ([0, 1], [0.2]),
        ([0, 1], [0.1, np.nan]),
        ([0, 1], [0.1, 2]),
    ],
)
def test_invalid_metric_inputs_rejected(labels, scores, policy):
    with pytest.raises(ValueError):
        metrics(labels, scores, 0.5, policy)
