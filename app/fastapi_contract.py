"""Canonical trusted recognition-event contract primitives."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


EventState = Literal[
    "DETECTED",
    "QUALITY_REJECTED",
    "LIVENESS_REJECTED",
    "RECOGNITION_CANDIDATE",
    "RECOGNITION_CONFIRMED",
    "UNKNOWN",
    "EXPIRED",
]


class RecognitionEvent(BaseModel):
    event_id: UUID
    camera_id: str = Field(min_length=1, max_length=128)
    device_id: str = Field(min_length=1, max_length=128)
    track_id: str = Field(min_length=1, max_length=128)
    state: EventState
    person_id: str | None = Field(default=None, max_length=128)
    model_version: str | None = Field(default=None, max_length=128)
    embedding_version: str | None = Field(default=None, max_length=128)
    quality: float | None = Field(default=None, ge=0.0, le=1.0)
    liveness: float | None = Field(default=None, ge=0.0, le=1.0)
    similarity: float | None = Field(default=None, ge=-1.0, le=1.0)
    margin: float | None = Field(default=None, ge=0.0)
    frames_confirmed: int = Field(default=0, ge=0)
    timestamp: datetime

    @field_validator("timestamp")
    @classmethod
    def require_aware_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp must include timezone")
        return value.astimezone(timezone.utc)

    @field_validator("person_id", "model_version", "embedding_version", "quality", "liveness", "similarity", "margin")
    @classmethod
    def confirmed_fields_required(cls, value, info):
        if info.data.get("state") == "RECOGNITION_CONFIRMED" and value is None:
            raise ValueError(f"{info.field_name} is required for confirmed events")
        return value

    @field_validator("frames_confirmed")
    @classmethod
    def confirmed_frames_required(cls, value, info):
        if info.data.get("state") == "RECOGNITION_CONFIRMED" and value < 1:
            raise ValueError("frames_confirmed must be >= 1 for confirmed events")
        return value
