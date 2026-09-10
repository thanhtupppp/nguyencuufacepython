"""MQTT transport adapter for already-authorized action envelopes.

The adapter intentionally depends on a tiny injected MQTT client protocol so the
core package does not require a broker client merely to validate event semantics.
"""
from __future__ import annotations

import json
from typing import Protocol

from .action_envelope import ActionEnvelope
from .transport import validate_action_envelope


class MqttPublisher(Protocol):
    def publish(self, topic: str, payload: str, qos: int, retain: bool):
        ...


class MqttActionTransport:
    """Publish ActionEnvelope without changing recognition semantics."""

    def __init__(self, publisher: MqttPublisher, device_id: str, qos: int = 1):
        if not device_id:
            raise ValueError("device_id is required")
        if qos not in (0, 1, 2):
            raise ValueError("qos must be 0, 1 or 2")
        self._publisher = publisher
        self.device_id = device_id
        self.qos = qos
        self.topic = f"face/{device_id}/actions/v1"

    def send(self, envelope: ActionEnvelope) -> str:
        validate_action_envelope(envelope)
        payload = json.dumps(envelope.to_dict(), sort_keys=True, separators=(",", ":"))
        result = self._publisher.publish(
            self.topic,
            payload,
            qos=self.qos,
            retain=False,
        )
        return getattr(result, "mid", envelope.event_id)
