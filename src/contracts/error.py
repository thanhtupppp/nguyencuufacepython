"""Sanitized error envelope shared by HTTP, WebSocket and action events."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from src.observability.request_context import get_request_id, now_utc

ErrorCode = Literal[
    "VALIDATION_ERROR",
    "DEPENDENCY_UNAVAILABLE",
    "AUTHORIZATION_FAILURE",
    "INTERNAL_ERROR",
    "MODEL_NOT_READY",
    "PROVIDER_UNAVAILABLE",
    "INFERENCE_SMOKE_FAILED",
    "PROVENANCE_INVALID",
    "DATASET_MANIFEST_INVALID",
    "LEDGER_UNAVAILABLE",
    "ACTION_DELIVERY_FAILED",
]


class ErrorEnvelope(BaseModel):
    code: ErrorCode
    message: str = Field(min_length=1, max_length=512)
    request_id: str = Field(min_length=1, max_length=128)
    timestamp: str


def error_envelope(code: ErrorCode, message: str) -> dict[str, str]:
    """Return a public-safe envelope; callers must never pass exception text."""
    return ErrorEnvelope(
        code=code,
        message=message,
        request_id=get_request_id() or "unknown",
        timestamp=now_utc(),
    ).model_dump()
