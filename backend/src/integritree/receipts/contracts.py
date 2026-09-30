"""Internal receipt confirmation contracts; not an upload or HTTP authorization layer.

The future image service must create ReceiptSource after validating a stored upload
and its supported layout. Never accept that trusted context from a client verbatim.
"""
from datetime import date, time
from decimal import Decimal
from hashlib import sha256
import json
import math
import re
from typing import Annotated, Literal
from uuid import UUID

from pydantic import ConfigDict, Field, StrictBool, StringConstraints, field_validator, model_validator

from integritree.contracts import Contract, Sha256

MAPPING_VERSION = "gcash_confirmed_v1"
Workflow = Literal["express_send", "pay_online", "bank_transfer"]
Role = Literal["personal_wallet", "merchant", "bank_account", "agent"]
Text = Annotated[str, StringConstraints(strict=True, min_length=1, max_length=256)]
OptionalText = Text | None
RULES = {
    "express_send": ("TRANSFER", "personal_wallet"),
    "pay_online": ("PAYMENT", "merchant"),
    "bank_transfer": ("DEBIT", "bank_account"),
}
LAYOUT_WORKFLOWS = {
    "gcash_express_send_v1": "express_send",
    "gcash_pay_online_v1": "pay_online",
    "gcash_bank_transfer_v1": "bank_transfer",
}


class ReceiptContract(Contract):
    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always",
                              allow_inf_nan=False, hide_input_in_errors=True)


class ExtractedReceiptFields(ReceiptContract):
    """Untrusted OCR candidates, including invalid text, retained for comparison."""
    workflow: OptionalText = None
    category: OptionalText = None
    amount: OptionalText = None
    date: OptionalText = None
    time: OptionalText = None
    reference: OptionalText = None
    origin_role: OptionalText = None
    destination_role: OptionalText = None
    origin_name: OptionalText = None
    destination_name: OptionalText = None


class ReceiptSource(ReceiptContract):
    """Server-owned upload/layout evidence; IDs are independent of payment reference.

    Layout IDs identify the initial mapping targets, not a claim that a production
    layout validator already exists. Unknown/QR/cash layouts cannot enter this path.
    """
    analysis_id: UUID
    image_id: UUID
    image_sha256: Sha256
    media_type: Literal["image/png", "image/jpeg"]
    layout_id: Literal["gcash_express_send_v1", "gcash_pay_online_v1", "gcash_bank_transfer_v1"]
    extractor_version: Text
    parser_version: Text
    extracted: ExtractedReceiptFields
    observed_destination_role: Role | None = None


class ConfirmedReceiptFields(ReceiptContract):
    workflow: Workflow
    category: Literal["TRANSFER", "PAYMENT", "DEBIT"]
    amount: Decimal
    date: str
    time: str
    currency: Literal["PHP"] = "PHP"
    timezone: Literal["Asia/Manila"] = "Asia/Manila"
    origin_role: Literal["personal_wallet"]
    destination_role: Role
    wallet_funded: StrictBool
    reference: OptionalText = None
    origin_name: OptionalText = None
    destination_name: OptionalText = None

    @field_validator("amount", mode="before")
    @classmethod
    def principal_decimal(cls, value):
        # No float coercion, exponent, grouping, currency symbol, or silent rounding.
        if not isinstance(value, (str, Decimal)):
            raise ValueError("principal_requires_decimal_string")
        text = str(value)
        if len(text) > 32 or not re.fullmatch(r"[0-9]+(?:\.[0-9]{1,2})?", text):
            raise ValueError("principal_requires_nonnegative_decimal_with_at_most_two_places")
        amount = Decimal(text)
        numeric = float(amount)
        if not math.isfinite(numeric) or Decimal(str(numeric)) != amount:
            raise ValueError("principal_not_representable_at_model_numeric_precision")
        # Apply the size guard to canonical output too, so nested validation/replay agree.
        canonical = format(amount, ".2f")
        if len(canonical) > 32:
            raise ValueError("principal_exceeds_canonical_length_limit")
        return Decimal(canonical)

    @field_validator("date", mode="before")
    @classmethod
    def calendar_date(cls, value):
        if not isinstance(value, str) or not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value):
            raise ValueError("date_requires_YYYY_MM_DD")
        try:
            date.fromisoformat(value)
        except ValueError:
            raise ValueError("invalid_calendar_date") from None
        return value

    @field_validator("time", mode="before")
    @classmethod
    def local_time(cls, value):
        if not isinstance(value, str) or not re.fullmatch(r"[0-9]{2}:[0-9]{2}(?::[0-9]{2})?", value):
            raise ValueError("time_requires_24_hour_HH_MM_or_HH_MM_SS")
        try:
            time.fromisoformat(value)
        except ValueError:
            raise ValueError("invalid_local_time") from None
        return value

    @field_validator("reference", "origin_name", "destination_name")
    @classmethod
    def nonblank_optional_text(cls, value):
        if value is not None and not value.strip():
            raise ValueError("use_null_for_unavailable_text")
        return value

    @field_validator("wallet_funded")
    @classmethod
    def require_wallet_funding(cls, value):
        if value is not True:
            raise ValueError("wallet_funding_required")
        return value

    @model_validator(mode="after")
    def compatible_mapping(self):
        category, destination = RULES[self.workflow]
        if self.category != category:
            raise ValueError("workflow_category_mismatch")
        if self.destination_role != destination:
            raise ValueError("workflow_destination_role_mismatch")
        return self


class ConfirmedReceipt(ReceiptContract):
    source: ReceiptSource
    fields: ConfirmedReceiptFields
    confirmed: StrictBool
    revision: Annotated[int, Field(strict=True, ge=1)]
    mapping_version: Literal["gcash_confirmed_v1"] = MAPPING_VERSION

    @model_validator(mode="after")
    def source_matches_confirmation(self):
        if self.confirmed is not True:
            raise ValueError("explicit_confirmation_required")
        if self.fields.workflow != LAYOUT_WORKFLOWS[self.source.layout_id]:
            raise ValueError("confirmed_workflow_does_not_match_supported_image_layout")
        observed = self.source.observed_destination_role
        if observed is not None and observed != self.fields.destination_role:
            raise ValueError("destination_role_conflicts_with_image_evidence")
        return self

    @property
    def input_sha256(self) -> str:
        serialized = json.dumps(self.model_dump(mode="json"), sort_keys=True,
                                separators=(",", ":"), ensure_ascii=False)
        return sha256(serialized.encode("utf-8")).hexdigest()

    @property
    def field_provenance(self) -> dict:
        """Literal candidate/confirmed comparison; normalized formatting is a correction."""
        confirmed = self.fields.model_dump(mode="json")
        result = {}
        for name, extracted in self.source.extracted.model_dump().items():
            value = confirmed[name]
            status = ("unchanged" if extracted == value else
                      "completed" if extracted is None else "corrected")
            if value is None and extracted is None:
                status = "unavailable"
            result[name] = {"extracted": extracted, "confirmed": value, "status": status}
        return result


def confirm_receipt(source: ReceiptSource, fields: ConfirmedReceiptFields, *,
                    confirmed: bool, previous: ConfirmedReceipt | None = None) -> ConfirmedReceipt:
    """Create a new immutable revision. The service must retire results for older revisions."""
    source = ReceiptSource.model_validate(source)
    revision = 1
    if previous is not None:
        previous = ConfirmedReceipt.model_validate(previous)
        if previous.source != source:
            raise ValueError("cannot_revise_a_different_receipt_source")
        revision = previous.revision + 1
    return ConfirmedReceipt(source=source, fields=fields, confirmed=confirmed, revision=revision)
