"""Transport adapter contract shared by MQTT, HTTP and WebSocket."""
from __future__ import annotations

from typing import Protocol

from .action_envelope import ActionEnvelope


class ActionTransport(Protocol):
    """Deliver an already-authorized envelope without changing its identity."""

    def send(self, envelope: ActionEnvelope) -> str:
        """Return the transport receipt/reference for the delivery attempt."""
        ...


def validate_action_envelope(envelope: ActionEnvelope) -> None:
    if envelope.schema_version != 1:
        raise ValueError("unsupported action envelope schema_version")
    if envelope.authorization_state != "AUTHORIZED":
        raise ValueError("only AUTHORIZED envelopes may enter action transport")
    if not envelope.event_id:
        raise ValueError("event_id is required")
    if not envelope.camera_id or not envelope.track_id or not envelope.person_id:
        raise ValueError("camera_id, track_id and person_id are required")
    if envelope.frame_index < 0:
        raise ValueError("frame_index must be >= 0")
