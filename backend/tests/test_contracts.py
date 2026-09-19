from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from integritree.api.schemas import PredictionRequest
from integritree.contracts import (
    ExperimentMetadata, ModelResult, PairedPrediction, PaySimRecord,
    PredictorInput, ShapExplanation, SourceRecordIdentity,
)


def test_raw_record_projects_only_allowed_predictor_fields(raw_record):
    record = PaySimRecord.model_validate(raw_record)
    projected = record.predictor_input().model_dump()
    assert set(projected) == {"step", "type", "amount", "nameOrig", "nameDest"}
    raw_record["isFraud"] = 0
    raw_record["oldbalanceOrg"] = 999.0
    assert PaySimRecord.model_validate(raw_record).predictor_input().model_dump() == projected


@pytest.mark.parametrize("forbidden", ["isFraud", "isFlaggedFraud", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"])
def test_prediction_contract_rejects_labels_and_balances(raw_record, forbidden):
    fields = PaySimRecord.model_validate(raw_record).predictor_input().model_dump()
    with pytest.raises(ValidationError, match=forbidden):
        PredictorInput.model_validate(fields | {forbidden: 1})


@pytest.mark.parametrize("field,value", [
    ("step", 0), ("step", 1.5), ("step", True),
    ("amount", -1), ("amount", float("inf")), ("amount", float("nan")),
    ("amount", True), ("amount", "100"),
    ("type", "UNKNOWN"), ("nameOrig", "  "), ("isFraud", 2), ("isFraud", True),
])
def test_bad_records_rejected(raw_record, field, value):
    with pytest.raises(ValidationError, match=field):
        PaySimRecord.model_validate(raw_record | {field: value})


def test_zero_amount_allowed(raw_record):
    assert PaySimRecord.model_validate(raw_record | {"amount": 0}).amount == 0


def test_identity_stable_and_distinct():
    first = SourceRecordIdentity(dataset_sha256="a" * 64, source_row_number=1)
    assert first.transaction_id == "a" * 64 + ":1"
    assert first.transaction_id == SourceRecordIdentity(**first.model_dump()).transaction_id
    assert first.transaction_id != SourceRecordIdentity(dataset_sha256="a" * 64, source_row_number=2).transaction_id
    assert first.transaction_id != SourceRecordIdentity(dataset_sha256="b" * 64, source_row_number=1).transaction_id
    with pytest.raises(ValidationError):
        SourceRecordIdentity(dataset_sha256="a" * 64, source_row_number=0)


def test_paired_results_allow_disagreement_but_reject_swapped_models():
    rf = ModelResult(model="rf", predicted_label=0, risk_score=0.2)
    smote = ModelResult(model="rf_smote", predicted_label=1, risk_score=0.8)
    pair = PairedPrediction(transaction_id="example", run_id="run", rf=rf, rf_smote=smote)
    assert pair.rf.predicted_label != pair.rf_smote.predicted_label
    assert "actual_label" not in pair.model_dump()
    with pytest.raises(ValidationError, match="match"):
        PairedPrediction(transaction_id="example", run_id="run", rf=smote, rf_smote=rf)


@pytest.mark.parametrize("value", [-0.01, 1.01, float("nan"), float("inf"), True, "0.5"])
def test_invalid_risk_score_rejected(value):
    with pytest.raises(ValidationError, match="risk_score"):
        ModelResult(model="rf", predicted_label=0, risk_score=value)


def test_prediction_envelope_rejects_ground_truth(raw_record):
    transaction = PaySimRecord.model_validate(raw_record).predictor_input()
    with pytest.raises(ValidationError, match="actual_label"):
        PredictionRequest(transaction_id="example", transaction=transaction, actual_label=1)


def test_shap_features_unique():
    feature = {"feature": "log_amount", "value": 1.0, "contribution": 0.1}
    with pytest.raises(ValidationError, match="unique"):
        ShapExplanation(output_space="probability", base_value=0.1, output_value=0.3, features=[feature, feature])


def test_metadata_requires_timezone_and_unique_features(draft):
    values = {
        "run_id": "test", "experiment_name": "test",
        "created_at": datetime.now(timezone.utc), "dataset": draft["dataset"],
        "configuration_sha256": "b" * 64, "split_manifest_sha256": "c" * 64,
        "feature_order": ["log_amount"], "package_versions": {"example": "1"},
    }
    assert ExperimentMetadata(**values).feature_order == ["log_amount"]
    with pytest.raises(ValidationError, match="timezone"):
        ExperimentMetadata(**(values | {"created_at": datetime(2026, 1, 1)}))
    with pytest.raises(ValidationError, match="duplicates"):
        ExperimentMetadata(**(values | {"feature_order": ["log_amount", "log_amount"]}))
