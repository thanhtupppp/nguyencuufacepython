"""HTTP transport adapter for already-authorized action envelopes."""
from __future__ import annotations

import json
from typing import Protocol

from .action_envelope import ActionEnvelope
from .transport import validate_action_envelope


class HttpPoster(Protocol):
    def post_json(self, url: str, payload: str, headers: dict[str, str]) -> str:
        ...


class HttpActionTransport:
    """Send an ActionEnvelope over HTTP without changing identity semantics."""

    def __init__(self, poster: HttpPoster, url: str) -> None:
        if not url:
            raise ValueError("url is required")
        self._poster = poster
        self.url = url

    def send(self, envelope: ActionEnvelope) -> str:
        validate_action_envelope(envelope)
        payload = json.dumps(envelope.to_dict(), sort_keys=True, separators=(",", ":"))
        return self._poster.post_json(
            self.url,
            payload,
            {"Content-Type": "application/json", "Idempotency-Key": envelope.event_id},
        )
