"""Confirmation, provenance and saved-model parity using synthetic records only."""
from datetime import date, timedelta
from decimal import Decimal
from types import SimpleNamespace
from uuid import UUID

import joblib
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from pydantic import ValidationError
import pytest
from sklearn.ensemble import RandomForestClassifier

from integritree.ml.features import FEATURE_COLUMNS, SCALED_COLUMNS, engineer_features
from integritree.ml.inference import predict_records
from integritree.ml.preprocessing import FittedPreprocessor
from integritree.receipts.contracts import (
    ConfirmedReceipt, ConfirmedReceiptFields, ExtractedReceiptFields, ReceiptSource,
    confirm_receipt,
)
from integritree.receipts.mapping import predict_receipt, receipt_features

WORKFLOWS = [
    ("express_send", "TRANSFER", "personal_wallet"),
    ("pay_online", "PAYMENT", "merchant"),
    ("bank_transfer", "DEBIT", "bank_account"),
]


def fields(**changes):
    return ConfirmedReceiptFields.model_validate({
        "workflow": "express_send", "category": "TRANSFER", "amount": "1234.50",
        "date": "2026-10-05", "time": "13:42", "origin_role": "personal_wallet",
        "destination_role": "personal_wallet", "wallet_funded": True,
    } | changes)


def source(workflow="express_send", **changes):
    return ReceiptSource.model_validate({
        "analysis_id": UUID(int=1), "image_id": UUID(int=2), "image_sha256": "a" * 64,
        "media_type": "image/jpeg", "layout_id": f"gcash_{workflow}_v1",
        "extractor_version": "synthetic_ocr_v1", "parser_version": "synthetic_parser_v1",
        "extracted": ExtractedReceiptFields(amount="123.45", reference="000123"),
    } | changes)


def receipt(**changes):
    values = fields(**changes)
    return confirm_receipt(source(values.workflow), values, confirmed=True)


@pytest.mark.parametrize("field,value", [
    ("amount", None), ("amount", True), ("amount", 1), ("amount", 1.25),
    ("amount", "-1.00"), ("amount", "NaN"), ("amount", "Infinity"),
    ("amount", "1e3"), ("amount", "1,000.00"), ("amount", "PHP 10.00"),
    ("amount", "1.001"), ("amount", " 10.00 "),
    ("amount", "9007199254740993.00"), ("amount", "1234567890123456.78"),
    ("amount", "1" + "0" * 29),
    ("date", None), ("date", "2026-02-29"), ("date", "2026-04-31"),
    ("date", "2026-1-1"), ("date", "2026-01-01T00:00:00"), ("date", 1),
    ("time", None), ("time", "1:00 PM"), ("time", "24:00"), ("time", "12:60"),
    ("time", "00:00+08:00"), ("time", "23:59:60"), ("time", "12:00:00.5"),
    ("currency", "USD"), ("timezone", "UTC"), ("reference", 123),
    ("reference", " "), ("origin_name", ""),
    ("origin_role", "merchant"), ("origin_role", "unknown"),
    ("destination_role", "unknown"), ("wallet_funded", False),
    ("wallet_funded", 1), ("wallet_funded", "true"),
    ("workflow", "merchant_qr"), ("workflow", "cash_in"), ("workflow", "cash_out"),
    ("category", "CASH_IN"), ("category", "CASH_OUT"),
    ("isFraud", 1), ("oldbalanceOrg", 100), ("step", 1),
    ("is_merchant_dest", 1), ("features", [0] * 11),
])
def test_strict_confirmed_fields_reject_invalid_missing_or_forbidden_values(field, value):
    with pytest.raises(ValidationError):
        fields(**{field: value})


@pytest.mark.parametrize("missing", ["amount", "date", "time", "wallet_funded", "origin_role", "destination_role"])
def test_required_fields_have_no_imputation(missing):
    payload = fields().model_dump()
    del payload[missing]
    with pytest.raises(ValidationError):
        ConfirmedReceiptFields.model_validate(payload)


@pytest.mark.parametrize("workflow,category,destination", WORKFLOWS)
def test_only_compatible_category_and_role_are_accepted(workflow, category, destination):
    values = fields(workflow=workflow, category=category, destination_role=destination)
    assert values.destination_role == destination
    for wrong in {"personal_wallet", "merchant", "bank_account", "agent"} - {destination}:
        with pytest.raises(ValidationError, match="workflow_destination_role_mismatch"):
            fields(workflow=workflow, category=category, destination_role=wrong)
    wrong_category = "PAYMENT" if category != "PAYMENT" else "TRANSFER"
    with pytest.raises(ValidationError, match="workflow_category_mismatch"):
        fields(workflow=workflow, category=wrong_category, destination_role=destination)


@pytest.mark.parametrize("confirmation", [False, 1, "true", None])
def test_explicit_boolean_confirmation_required(confirmation):
    with pytest.raises(ValidationError):
        confirm_receipt(source(), fields(), confirmed=confirmation)


def test_image_context_and_compatible_observed_role_are_required():
    with pytest.raises(ValidationError):
        ConfirmedReceipt(fields=fields(), confirmed=True, revision=1)
    with pytest.raises(ValidationError):
        source(layout_id="unresolved")
    with pytest.raises(ValidationError):
        source(media_type="application/pdf")
    with pytest.raises(ValidationError):
        source(image_sha256="invalid")
    with pytest.raises(ValidationError, match="image_layout"):
        confirm_receipt(source("pay_online"), fields(), confirmed=True)
    # Bank Transfer heading with a known wallet destination cannot become DEBIT.
    bank_fields = fields(workflow="bank_transfer", category="DEBIT", destination_role="bank_account")
    with pytest.raises(ValidationError, match="image_evidence"):
        confirm_receipt(source("bank_transfer", observed_destination_role="personal_wallet"),
                        bank_fields, confirmed=True)


def test_decimal_json_roundtrip_reference_and_calendar_boundaries():
    item = receipt(amount="0", date="2024-02-29", time="00:00:01", reference="0000123")
    assert item.fields.amount == Decimal("0.00")
    assert item.fields.model_dump(mode="json")["amount"] == "0.00"
    assert item.fields.reference == "0000123"
    assert ConfirmedReceipt.model_validate_json(item.model_dump_json()) == item
    large = receipt(amount="1" + "0" * 28)
    assert ConfirmedReceipt.model_validate_json(large.model_dump_json()) == large
    features = receipt_features(item).iloc[0]
    assert features.hour_of_day == 0 and features.day_of_week == 3
    assert features.is_zero_amount == 1 and features.log_amount == 0
    sunday = receipt_features(receipt(date="2026-10-11", time="23:59:59")).iloc[0]
    assert sunday.hour_of_day == 23 and sunday.day_of_week == 6


def test_provenance_and_revisions_bind_source_and_preserve_optional_fields():
    item = receipt(reference="000123")
    assert item.field_provenance["amount"] == {
        "extracted": "123.45", "confirmed": "1234.50", "status": "corrected"}
    assert item.field_provenance["date"]["status"] == "completed"
    assert item.field_provenance["reference"]["status"] == "unchanged"
    assert item.field_provenance["origin_name"]["status"] == "unavailable"
    revised = confirm_receipt(item.source, fields(amount="0"), confirmed=True, previous=item)
    assert revised.revision == 2 and item.revision == 1
    assert revised.input_sha256 != item.input_sha256
    assert revised.field_provenance["reference"]["status"] == "corrected"
    with pytest.raises(ValueError, match="different_receipt_source"):
        confirm_receipt(source(image_id=UUID(int=3)), fields(), confirmed=True, previous=item)
    with pytest.raises(ValidationError):
        item.fields.amount = Decimal("99")


def test_model_copy_cannot_bypass_mapping_validation():
    item = receipt()
    unsafe = item.model_copy(update={"confirmed": False})
    with pytest.raises(ValidationError):
        receipt_features(unsafe)
    unsafe_fields = item.fields.model_copy(update={"destination_role": "merchant"})
    with pytest.raises(ValidationError):
        receipt_features(item.model_copy(update={"fields": unsafe_fields}))


def test_names_reference_and_ids_do_not_change_features():
    first = receipt()
    other = confirm_receipt(source(analysis_id=UUID(int=4)), fields(
        origin_name="M_SYNTHETIC_NAME", destination_name="M_ANOTHER_NAME", reference="0099"),
        confirmed=True)
    assert_frame_equal(receipt_features(first), receipt_features(other))
    assert first.input_sha256 != other.input_sha256


def synthetic_raw(day, hour, category, amount):
    # Test-only equivalence witness. Production receipt mapping never fabricates step/IDs.
    return {"step": day * 24 + hour + 1, "type": category, "amount": float(amount),
            "nameOrig": "C_SYNTHETIC", "nameDest": "M_SYNTHETIC" if category == "PAYMENT" else "C_SYNTHETIC_DEST"}


@pytest.fixture
def saved_bundle(tmp_path):
    training = pd.DataFrame([synthetic_raw(i % 7, i % 24, WORKFLOWS[i % 3][1], str(i))
                             for i in range(48)])
    preprocessor = FittedPreprocessor.fit(training)
    preprocessor.save(tmp_path / "preprocessing.json")
    matrix = preprocessor.transform(training).to_numpy(dtype="float64")
    # Small synthetic models, not retraining retained research artifacts.
    models = {}
    for name, seed in [("rf", 42), ("rf_smote", 43)]:
        model = RandomForestClassifier(n_estimators=4, max_depth=3, random_state=seed).fit(
            matrix, np.arange(len(matrix)) % 2)
        joblib.dump(model, tmp_path / f"{name}.joblib")
        models[name] = joblib.load(tmp_path / f"{name}.joblib")
    return SimpleNamespace(models=models, preprocessor=FittedPreprocessor.load(tmp_path / "preprocessing.json"),
                           config=SimpleNamespace(scoring=SimpleNamespace(threshold=.43)),
                           metadata={"run_id": "synthetic_saved_pair"})


@pytest.mark.parametrize("workflow,category,destination", WORKFLOWS)
def test_all_weekday_hour_features_scaling_and_saved_predictions_match_raw_path(
        workflow, category, destination, saved_bundle):
    receipt_rows, raw_rows, items = [], [], []
    original_state = saved_bundle.preprocessor.state.model_dump_json()
    for day in range(7):
        for hour in range(24):
            amount = ["0.00", "0.01", "1234.50", "9999999.99"][hour % 4]
            item = receipt(workflow=workflow, category=category, destination_role=destination,
                           amount=amount, date=(date(2026, 10, 5) + timedelta(days=day)).isoformat(),
                           time=f"{hour:02d}:59")
            items.append(item)
            raw_rows.append(synthetic_raw(day, hour, category, amount))
            receipt_rows.append(receipt_features(item))
    raw = pd.DataFrame(raw_rows)
    unscaled = pd.concat(receipt_rows, ignore_index=True)
    assert_frame_equal(unscaled, engineer_features(raw, saved_bundle.preprocessor.state.amount_median))
    expected = saved_bundle.preprocessor.transform(raw)
    assert_frame_equal(saved_bundle.preprocessor.transform_engineered(unscaled), expected)
    # Include zero, fractional, out-of-training-range amounts and week/hour edges.
    for i in [0, 1, 23, 24, 167]:
        result = predict_receipt(saved_bundle, items[i])
        equivalent = predict_records(saved_bundle, raw.iloc[[i]], [str(items[i].source.analysis_id)])
        assert_frame_equal(result.predictions, equivalent)
        assert_frame_equal(result.scaled_features, expected.iloc[[i]].reset_index(drop=True))
        assert result.is_current(items[i])
        assert "actual_label" not in result.predictions
    assert expected.log_amount.max() > 1  # Saved scaling is not clipped or refit.
    assert saved_bundle.preprocessor.state.model_dump_json() == original_state
    for column in set(FEATURE_COLUMNS) - set(SCALED_COLUMNS):
        assert set(expected[column]).issubset({0, 1})


def test_corrected_input_replaces_features_and_marks_old_results_stale(saved_bundle):
    old = receipt(amount="1.00")
    result = predict_receipt(saved_bundle, old)
    corrected = confirm_receipt(old.source, fields(amount="2000.00"), confirmed=True, previous=old)
    replacement = predict_receipt(saved_bundle, corrected)
    assert not result.is_current(corrected)
    assert replacement.is_current(corrected)
    assert replacement.unscaled_features.log_amount.iloc[0] == np.log1p(2000.00)
    assert replacement.predictions.transaction_id.iloc[0] == str(old.source.analysis_id)
    assert replacement.predictions.run_id.iloc[0] == saved_bundle.metadata["run_id"]
    for name in ("rf", "rf_smote"):
        scores = replacement.predictions[f"{name}_risk_score"]
        np.testing.assert_array_equal(replacement.predictions[f"{name}_predicted_label"], scores >= .43)


def test_shared_scaling_rejects_wrong_schema_and_nonfinite_values_without_mutation(saved_bundle):
    raw = receipt_features(receipt())
    before = raw.copy(deep=True)
    saved_bundle.preprocessor.transform_engineered(raw)
    assert_frame_equal(raw, before)
    with pytest.raises(ValueError, match="columns/order"):
        saved_bundle.preprocessor.transform_engineered(raw.iloc[:, ::-1])
    raw.loc[0, "log_amount"] = np.inf
    with pytest.raises(ValueError, match="finite"):
        saved_bundle.preprocessor.transform_engineered(raw)


def test_saved_threshold_exact_tie_and_paired_model_order(saved_bundle):
    class FixedModel:
        classes_ = np.array([1, 0])

        def __init__(self, score):
            self.score = score

        def predict_proba(self, matrix):
            return np.array([[self.score, 1 - self.score]] * len(matrix))

    saved_bundle.models = {"rf": FixedModel(.43), "rf_smote": FixedModel(.429999)}
    result = predict_receipt(saved_bundle, receipt()).predictions.iloc[0]
    assert result.rf_predicted_label == 1
    assert result.rf_smote_predicted_label == 0
    assert result.rf_risk_score == .43
