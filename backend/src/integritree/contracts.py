"""Shared data contracts. No model training or feature engineering occurs here."""

from typing import Annotated, Literal

from pydantic import (
    AwareDatetime, BaseModel, ConfigDict, Field, StringConstraints, model_validator,
)

Label = Annotated[int, Field(strict=True, ge=0, le=1)]
PositiveInteger = Annotated[int, Field(strict=True, gt=0)]
NonnegativeNumber = Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]
Probability = Annotated[float, Field(strict=True, ge=0, le=1, allow_inf_nan=False)]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
NonemptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
TransactionType = Literal["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
ModelName = Literal["rf", "rf_smote"]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class PredictorInput(Contract):
    """Raw source attributes for later feature engineering, never ground truth."""

    step: PositiveInteger
    type: TransactionType
    amount: NonnegativeNumber
    nameOrig: NonemptyText
    nameDest: NonemptyText


class PaySimRecord(PredictorInput):
    """All eleven CSV fields, after a CSV loader converts numeric text."""

    oldbalanceOrg: NonnegativeNumber
    newbalanceOrig: NonnegativeNumber
    oldbalanceDest: NonnegativeNumber
    newbalanceDest: NonnegativeNumber
    isFraud: Label
    isFlaggedFraud: Label

    def predictor_input(self) -> PredictorInput:
        return PredictorInput.model_validate(
            {name: getattr(self, name) for name in PredictorInput.model_fields}
        )


class SourceRecordIdentity(Contract):
    dataset_sha256: Sha256
    source_row_number: PositiveInteger

    @property
    def transaction_id(self) -> str:
        """One-based CSV data row before cleaning; the header is not a data row."""
        return f"{self.dataset_sha256}:{self.source_row_number}"


class LabeledTransaction(Contract):
    identity: SourceRecordIdentity
    transaction: PredictorInput
    actual_label: Label


class FeatureContribution(Contract):
    feature: NonemptyText
    value: float
    contribution: float


class ShapExplanation(Contract):
    output_space: Literal["probability", "raw"]
    base_value: float
    output_value: float
    features: list[FeatureContribution] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_features(self) -> "ShapExplanation":
        names = [item.feature for item in self.features]
        if len(names) != len(set(names)):
            raise ValueError("SHAP feature names must be unique")
        return self


class ModelResult(Contract):
    model: ModelName
    predicted_label: Label
    risk_score: Probability
    explanation: ShapExplanation | None = None


class PairedPrediction(Contract):
    transaction_id: NonemptyText
    run_id: NonemptyText
    rf: ModelResult
    rf_smote: ModelResult

    @model_validator(mode="after")
    def models_match_keys(self) -> "PairedPrediction":
        if self.rf.model != "rf" or self.rf_smote.model != "rf_smote":
            raise ValueError("Each result must match its rf or rf_smote key")
        return self


class DatasetIdentity(Contract):
    filename: NonemptyText
    source_url: NonemptyText
    sha256: Sha256
    row_count: PositiveInteger


class ExperimentMetadata(Contract):
    run_id: NonemptyText
    experiment_name: NonemptyText
    created_at: AwareDatetime
    dataset: DatasetIdentity
    configuration_sha256: Sha256
    split_manifest_sha256: Sha256
    feature_order: list[NonemptyText] = Field(min_length=1)
    package_versions: dict[str, str] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_feature_order(self) -> "ExperimentMetadata":
        if len(self.feature_order) != len(set(self.feature_order)):
            raise ValueError("feature_order must not contain duplicates")
        return self
