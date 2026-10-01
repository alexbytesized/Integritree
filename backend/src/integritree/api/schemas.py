"""API envelopes around the shared, framework-independent contracts."""

from typing import Literal

from integritree.contracts import Contract


class HealthResponse(Contract):
    status: Literal["ok"] = "ok"
    service: Literal["integritree-backend"] = "integritree-backend"
    version: str
