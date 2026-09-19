"""Hand-worked features, train-only fitting, and replay at the inference boundary."""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from integritree.ml.features import (
    FEATURE_COLUMNS, SOURCE_COLUMNS, SCALED_COLUMNS, DataValidationError,
    engineer_features,
)
from integritree.ml.preprocessing import FittedPreprocessor


def inputs(amounts=(0.0, 99.0)):
    return pd.DataFrame({
        "step": [1, 169], "type": ["TRANSFER", "PAYMENT"], "amount": amounts,
        "nameOrig": ["C1", "M2"], "nameDest": ["C3", "M4"],
    })


def test_hand_worked_features_and_week_wrap():
    frame = inputs()
    values = engineer_features(frame, 49.5)
    assert list(values.columns) == FEATURE_COLUMNS
    assert values["hour_of_day"].tolist() == [0, 0]
    assert values["day_of_week"].tolist() == [0, 0]
    assert values["is_zero_amount"].tolist() == [1, 0]
    np.testing.assert_allclose(values["log_amount"], [0, np.log(100)])
    assert values["is_merchant_origin"].tolist() == [0, 1]
    assert values["is_merchant_dest"].tolist() == [0, 1]
    frame["step"] = [24, 25]
    values = engineer_features(frame, 0)
    assert values["hour_of_day"].tolist() == [23, 0]
    assert values["day_of_week"].tolist() == [0, 1]
    assert values["type_TRANSFER"].tolist() == [1, 0]


def test_only_training_values_control_median_scaling_and_no_clipping(tmp_path):
    train = inputs()
    train["step"] = [1, 26]
    fitted = FittedPreprocessor.fit(train)
    original = fitted.state.model_dump()
    test = inputs((np.nan, 9999.0))
    test["step"] = [50, 100]
    transformed = fitted.transform(test)
    assert fitted.state.amount_median == 49.5
    assert transformed.iloc[0]["log_amount"] == pytest.approx(np.log1p(49.5) / np.log(100))
    assert transformed.iloc[1]["log_amount"] == pytest.approx(2)
    assert transformed.iloc[1]["hour_of_day"] == 3
    assert fitted.state.model_dump() == original
    for column in set(FEATURE_COLUMNS) - set(SCALED_COLUMNS):
        assert set(transformed[column]).issubset({0, 1})
    path = tmp_path / "preprocessing.json"
    fitted.save(path)
    replay = FittedPreprocessor.load(path)
    assert_frame_equal(replay.transform(test), transformed)
    individual = pd.concat([replay.transform(test.iloc[[i]]) for i in range(len(test))])
    assert_frame_equal(individual, transformed)


def test_batched_fitting_matches_whole_training_set():
    frame = inputs()
    frame["step"] = [1, 26]
    whole = FittedPreprocessor.fit(frame)
    batched = FittedPreprocessor.fit_batches(49.5, [frame.iloc[:1], frame.iloc[1:]])
    assert whole.state.model_dump() == batched.state.model_dump()


def test_constant_columns_and_unknown_type():
    frame = inputs((10.0, 10.0))
    frame["type"] = ["", None]
    fitted = FittedPreprocessor.fit(frame)
    transformed = fitted.transform(frame)
    assert transformed[SCALED_COLUMNS].eq(0).all().all()
    assert transformed.filter(like="type_").eq(0).all().all()


@pytest.mark.parametrize("column,value", [
    ("step", None), ("step", 1.5), ("step", True),
    ("amount", -1), ("amount", "bad"), ("amount", np.inf),
    ("nameOrig", None), ("nameDest", "X1"), ("type", "TRASFER"),
])
def test_invalid_predictor_data_rejected(column, value):
    frame = inputs().astype(object)
    frame.loc[0, column] = value
    with pytest.raises(DataValidationError, match=column):
        engineer_features(frame, 1.0)


@pytest.mark.parametrize("extra", ["isFraud", "oldbalanceOrg", "source_row_number"])
def test_extra_fields_never_enter_preprocessor(extra):
    frame = inputs()
    frame[extra] = 1
    with pytest.raises(ValueError, match="exactly"):
        FittedPreprocessor.fit(frame)


def test_no_training_amount_information_fails():
    with pytest.raises(ValueError, match="entirely missing"):
        FittedPreprocessor.fit(inputs((np.nan, np.nan)))


def test_invalid_saved_state_rejected(tmp_path):
    fitted = FittedPreprocessor.fit(inputs())
    path = tmp_path / "preprocessing.json"
    fitted.save(path)
    path.write_text(path.read_text().replace('"clip": false', '"clip": true'))
    with pytest.raises(ValueError):
        FittedPreprocessor.load(path)
