"""Transport-neutral action envelope for authorized face events."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

from .authorization_event import authorization_event_id


@dataclass(frozen=True)
class ActionEnvelope:
    schema_version: int
    event_id: str
    camera_id: str
    frame_index: int
    track_id: str
    person_id: str
    authorization_state: str
    timestamp: str

    @classmethod
    def authorized(
        cls,
        *,
        camera_id: str,
        frame_index: int,
        track_id: str,
        person_id: str,
        timestamp: str,
        schema_version: int = 1,
    ) -> "ActionEnvelope":
        if not timestamp:
            raise ValueError("timestamp is required")
        event_id = authorization_event_id(
            camera_id=camera_id,
            frame_index=frame_index,
            track_id=track_id,
            person_id=person_id,
        )
        return cls(
            schema_version=schema_version,
            event_id=event_id,
            camera_id=camera_id,
            frame_index=frame_index,
            track_id=track_id,
            person_id=person_id,
            authorization_state="AUTHORIZED",
            timestamp=timestamp,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json_dict(self) -> dict[str, Any]:
        return self.to_dict()
