import os
from pathlib import Path
import sys
import uuid

import numpy as np
import pandas as pd
import pytest

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
os.environ["MPLCONFIGDIR"] = str(ROOT / "runtime" / "matplotlib")
os.environ["MPLBACKEND"] = "Agg"

from integritree.config import load_experiment


def pytest_configure(config):
    # Use a fresh owned directory; do not depend on access to Windows' shared temp.
    if config.option.basetemp is None:
        (ROOT / "runtime" / "pytest").mkdir(parents=True, exist_ok=True)
        config.option.basetemp = str(ROOT / "runtime" / "pytest" / uuid.uuid4().hex)


@pytest.fixture
def config():
    return load_experiment(ROOT.parent / "configs" / "experiment.yaml")


@pytest.fixture
def frame():
    codes = np.arange(8)
    return pd.DataFrame(
        {
            "source_row_number": codes + 1,
            "transaction_id": [f"hash:{i + 1}" for i in codes],
            "actual_label": codes // 4,
            "rf_predicted_label": codes // 2 % 2,
            "rf_smote_predicted_label": codes % 2,
            "rf_risk_score": np.where(codes // 2 % 2, 0.8, 0.2),
            "rf_smote_risk_score": np.where(codes % 2, 0.9, 0.1),
            "split": "validation",
            "run_id": "model",
            "threshold": 0.43,
            "type": ["TRANSFER", "CASH_OUT"] * 4,
            "amount": [
                0.0,
                999.0,
                1000.0,
                9999.0,
                10000.0,
                99999.0,
                100000.0,
                1000000.0,
            ],
            "step": [1, 24, 25, 168, 169, 170, 171, 172],
            "is_zero_amount": [1, 0, 0, 0, 0, 0, 0, 0],
            "is_merchant_origin": 0,
            "is_merchant_dest": 0,
        }
    )
