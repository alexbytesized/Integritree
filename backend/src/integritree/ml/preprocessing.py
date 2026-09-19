"""Training-fitted amount imputation and Min-Max scaling with portable JSON state."""

from collections.abc import Iterable
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
import pandas as pd
from pydantic import Field, model_validator
from sklearn.preprocessing import MinMaxScaler

from integritree.contracts import Contract, NonnegativeNumber, PositiveInteger
from integritree.ml.features import (
    FEATURE_COLUMNS, SCALED_COLUMNS, TIME_CONVENTION, engineer_features, validate_predictors,
)

Triple = Annotated[list[float], Field(min_length=3, max_length=3)]


class PreprocessingState(Contract):
    schema_version: Literal[1] = 1
    time_convention: Literal["simulation_step_1_hour_0_day_0"] = TIME_CONVENTION
    feature_order: list[str]
    scaled_columns: list[str]
    amount_median: NonnegativeNumber
    scale: Triple
    offset: Triple
    data_min: Triple
    data_max: Triple
    training_rows: PositiveInteger
    clip: Literal[False] = False

    @model_validator(mode="after")
    def consistent_state(self):
        if self.feature_order != FEATURE_COLUMNS or self.scaled_columns != SCALED_COLUMNS:
            raise ValueError("Unsupported preprocessing feature schema/order")
        low, high = np.array(self.data_min), np.array(self.data_max)
        if np.any(low < 0) or np.any(high < low) or np.any(np.array(self.scale) <= 0):
            raise ValueError("Invalid scaler extrema or scale")
        return self


class FittedPreprocessor:
    def __init__(self, state: PreprocessingState):
        self.state = state

    @classmethod
    def fit(cls, training_inputs: pd.DataFrame):
        inputs = validate_predictors(training_inputs)
        if not inputs["amount"].notna().any():
            raise ValueError("Training amount column is entirely missing")
        return cls.fit_batches(float(inputs["amount"].median()), [inputs])

    @classmethod
    def fit_batches(cls, amount_median: float, training_batches: Iterable[pd.DataFrame]):
        scaler = MinMaxScaler(clip=False)
        rows = 0
        for frame in training_batches:
            if frame.empty:
                continue
            engineered = engineer_features(frame, amount_median)
            scaler.partial_fit(engineered[SCALED_COLUMNS])
            rows += len(frame)
        if rows == 0:
            raise ValueError("Cannot fit preprocessing without training records")
        return cls(PreprocessingState(
            feature_order=FEATURE_COLUMNS, scaled_columns=SCALED_COLUMNS,
            amount_median=amount_median, scale=scaler.scale_.tolist(),
            offset=scaler.min_.tolist(), data_min=scaler.data_min_.tolist(),
            data_max=scaler.data_max_.tolist(), training_rows=rows,
        ))

    def transform(self, inputs: pd.DataFrame) -> pd.DataFrame:
        features = engineer_features(inputs, self.state.amount_median)
        # Exact MinMaxScaler transform (multiply then add), without refitting/clipping.
        scaled = features[SCALED_COLUMNS].to_numpy(dtype="float64", copy=True)
        scaled *= np.array(self.state.scale)
        scaled += np.array(self.state.offset)
        features[SCALED_COLUMNS] = scaled
        return features.loc[:, self.state.feature_order]

    def save(self, path: Path) -> None:
        path.write_text(self.state.model_dump_json(indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path):
        return cls(PreprocessingState.model_validate_json(path.read_text(encoding="utf-8")))
