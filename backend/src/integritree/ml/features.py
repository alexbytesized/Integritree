"""Approved deterministic features shared by research and later inference."""

import numpy as np
import pandas as pd

SOURCE_COLUMNS = ["step", "type", "amount", "nameOrig", "nameDest"]
TRANSACTION_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
FEATURE_COLUMNS = [
    "hour_of_day", "day_of_week", *[f"type_{t}" for t in TRANSACTION_TYPES],
    "log_amount", "is_zero_amount", "is_merchant_origin", "is_merchant_dest",
]
SCALED_COLUMNS = ["log_amount", "hour_of_day", "day_of_week"]
TIME_CONVENTION = "simulation_step_1_hour_0_day_0"


class DataValidationError(ValueError):
    """Aggregate errors carry row positions, never raw transaction contents."""

    def __init__(self, issues: dict):
        self.issues = issues
        super().__init__("Invalid transaction inputs: " + ", ".join(issues))


def validate_predictors(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize five source attributes, allowing only missing amount/type.

    This ingestion boundary intentionally precedes the complete-record contracts.
    No labels, balances, flags, or arbitrary additional fields are accepted.
    """
    if len(frame.columns) != len(SOURCE_COLUMNS) or set(frame.columns) != set(SOURCE_COLUMNS):
        raise ValueError("Predictor columns must be exactly: " + ", ".join(SOURCE_COLUMNS))
    result = frame.loc[:, SOURCE_COLUMNS].copy()
    issues = {}

    def report(name, mask):
        mask = np.asarray(mask, dtype=bool)
        if mask.any():
            issues[name] = {"count": int(mask.sum()), "row_positions": np.flatnonzero(mask)[:20].tolist()}

    for name in ["step", "amount"]:
        values = result[name].astype("string[pyarrow]").str.strip()
        missing = values.isna() | values.eq("").fillna(False)
        numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype="float64", na_value=np.nan)
        invalid = ~missing.to_numpy() & (~np.isfinite(numeric) | (numeric < 0))
        if name == "step":
            invalid |= missing.to_numpy() | (numeric < 1) | (numeric != np.floor(numeric)) | (numeric > 2**53 - 1)
        report(name, invalid)
        result[name] = numeric

    for name in ["nameOrig", "nameDest"]:
        values = result[name].astype("string[pyarrow]").str.strip()
        valid = values.str.match(r"^[CM].+$").fillna(False)
        report(name, ~valid)
        result[name] = values

    types = result["type"].astype("string[pyarrow]").str.strip()
    missing_type = types.isna() | types.eq("").fillna(False)
    types = types.mask(missing_type, "unknown")
    report("type", ~types.isin(TRANSACTION_TYPES + ["unknown"]))
    result["type"] = types
    if issues:
        raise DataValidationError(issues)
    result["step"] = result["step"].astype("int64")
    return result


def engineer_features(frame: pd.DataFrame, amount_median: float) -> pd.DataFrame:
    """No learned operations: apply the supplied training median and formulas."""
    if not np.isfinite(amount_median) or amount_median < 0:
        raise ValueError("A finite nonnegative training amount median is required")
    values = validate_predictors(frame)
    amount = values["amount"].fillna(amount_median)
    step = values["step"] - 1
    features = pd.DataFrame(index=frame.index)
    features["hour_of_day"] = step % 24
    features["day_of_week"] = (step // 24) % 7
    for category in TRANSACTION_TYPES:
        features[f"type_{category}"] = (values["type"] == category).astype("uint8")
    features["log_amount"] = np.log1p(amount)
    features["is_zero_amount"] = (amount == 0).astype("uint8")
    features["is_merchant_origin"] = values["nameOrig"].str.startswith("M").astype("uint8")
    features["is_merchant_dest"] = values["nameDest"].str.startswith("M").astype("uint8")
    return features.loc[:, FEATURE_COLUMNS]
