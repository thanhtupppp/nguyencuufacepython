"""WebSocket transport adapter for authorized action envelopes."""
from __future__ import annotations

import json
from typing import Protocol

from .action_envelope import ActionEnvelope
from .transport import validate_action_envelope


class WebSocketSender(Protocol):
    def send_text(self, payload: str) -> str:
        ...


class WebSocketActionTransport:
    """Send an ActionEnvelope over a connected WebSocket."""

    def __init__(self, sender: WebSocketSender) -> None:
        self._sender = sender

    def send(self, envelope: ActionEnvelope) -> str:
        validate_action_envelope(envelope)
        payload = json.dumps(envelope.to_dict(), sort_keys=True, separators=(",", ":"))
        return self._sender.send_text(payload)
