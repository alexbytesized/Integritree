"""Score meaning, exact ties, and strict model-input boundaries."""
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from integritree.ml.features import FEATURE_COLUMNS
from integritree.ml.inference import classify, fraud_scores, feature_matrix, predict_features


def test_probability_column_lookup_and_exact_ties():
    class ReversedModel:
        classes_ = np.array([1, 0])
        def predict_proba(self, values):
            return np.array([[.5, .5], [.49, .51], [.51, .49]])
    scores = fraud_scores(ReversedModel(), np.zeros((3, 11)))
    np.testing.assert_array_equal(classify(scores, .5), [1, 0, 1])


def test_mean_tree_probability_is_not_hard_vote_fraction():
    X = np.zeros((10, 11))
    y = [0] * 7 + [1] * 3
    model = RandomForestClassifier(n_estimators=3, bootstrap=False, random_state=42).fit(X, y)
    scores = fraud_scores(model, X[:1])
    assert scores[0] == pytest.approx(.3)
    assert np.mean([tree.predict(X[:1])[0] for tree in model.estimators_]) == 0


@pytest.mark.parametrize("change", ["label", "order", "nan", "empty"])
def test_invalid_feature_matrix_rejected(change):
    X = pd.DataFrame(np.zeros((2, 11)), columns=FEATURE_COLUMNS)
    if change == "label":
        X["isFraud"] = 0
    elif change == "order":
        X = X.iloc[:, ::-1]
    elif change == "nan":
        X.iloc[0, 0] = np.nan
    else:
        X = X.iloc[:0]
    with pytest.raises(ValueError):
        feature_matrix(X)


@pytest.mark.parametrize("ids", [["same", "same"], ["only_one"], ["", "two"]])
def test_identity_errors_rejected(ids):
    X = pd.DataFrame(np.zeros((2, 11)), columns=FEATURE_COLUMNS)
    with pytest.raises(ValueError, match="transaction ID|Transaction IDs"):
        predict_features({}, X, ids, .5, "run")
