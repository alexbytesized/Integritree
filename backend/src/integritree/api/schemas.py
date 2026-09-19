"""API envelopes around the shared, framework-independent contracts."""

from typing import Literal

from integritree.contracts import Contract, NonemptyText, PredictorInput


class HealthResponse(Contract):
    status: Literal["ok"] = "ok"
    service: Literal["integritree-backend"] = "integritree-backend"
    version: str


class PredictionRequest(Contract):
    """Reserved contract for Phase 5; no prediction route exists yet."""

    transaction_id: NonemptyText
    transaction: PredictorInput
