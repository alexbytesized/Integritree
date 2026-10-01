"""Versioned confirmed-receipt mapping; no image handling or fabricated PaySim IDs."""
from dataclasses import dataclass
from datetime import date, time

import numpy as np
import pandas as pd

from integritree.ml.features import FEATURE_COLUMNS, TRANSACTION_TYPES
from integritree.ml.inference import predict_features
from integritree.receipts.contracts import ConfirmedReceipt


def receipt_features(receipt: ConfirmedReceipt) -> pd.DataFrame:
    """Derive eleven unscaled features under the approved demonstration assumptions."""
    receipt = ConfirmedReceipt.model_validate(receipt)
    values = receipt.fields
    amount = float(values.amount)
    frame = pd.DataFrame({
        "hour_of_day": [time.fromisoformat(values.time).hour],
        "day_of_week": [date.fromisoformat(values.date).weekday()],
    })
    for category in TRANSACTION_TYPES:
        frame[f"type_{category}"] = np.array([values.category == category], dtype="uint8")
    frame["log_amount"] = np.log1p(np.array([amount], dtype="float64"))
    frame["is_zero_amount"] = np.array([amount == 0], dtype="uint8")
    frame["is_merchant_origin"] = np.array([values.origin_role == "merchant"], dtype="uint8")
    frame["is_merchant_dest"] = np.array([values.destination_role == "merchant"], dtype="uint8")
    return frame.loc[:, FEATURE_COLUMNS]


@dataclass(frozen=True)
class ReceiptPrediction:
    """Internal result snapshot; HTTP/export serialization is a later service concern."""
    receipt: ConfirmedReceipt
    input_sha256: str
    unscaled_features: pd.DataFrame
    scaled_features: pd.DataFrame
    predictions: pd.DataFrame

    def is_current(self, receipt: ConfirmedReceipt) -> bool:
        receipt = ConfirmedReceipt.model_validate(receipt)
        return self.input_sha256 == receipt.input_sha256


def predict_receipt(bundle, receipt: ConfirmedReceipt) -> ReceiptPrediction:
    """Use a trusted loaded bundle's saved scaling, both models and shared threshold.

    The future image service must supply upload-bound confirmation and invalidate
    stored results on revision. This internal function does not expose manual entry.
    """
    receipt = ConfirmedReceipt.model_validate(receipt)
    unscaled = receipt_features(receipt)
    scaled = bundle.preprocessor.transform_engineered(unscaled)
    predictions = predict_features(bundle.models, scaled, [str(receipt.source.analysis_id)],
                                   bundle.config.scoring.threshold, bundle.metadata["run_id"])
    return ReceiptPrediction(receipt, receipt.input_sha256, unscaled, scaled, predictions)
