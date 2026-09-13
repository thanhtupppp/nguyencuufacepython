"""Bounded MQTT action transport with topic/payload validation."""
from __future__ import annotations

import json
import time
from typing import Protocol

from .action_envelope import ActionEnvelope
from .transport import validate_action_envelope


class MqttPublisher(Protocol):
    def publish(self, topic: str, payload: str, qos: int, retain: bool): ...


class MqttActionTransport:
    """Publish only authorized actions with bounded retry and terminal outcomes."""

    def __init__(self, publisher: MqttPublisher, device_id: str, qos: int = 1, *, max_retries: int = 2, retry_delay_s: float = 0.05):
        if not device_id or "/" in device_id or "#" in device_id or "+" in device_id:
            raise ValueError("device_id is required and MQTT-topic safe")
        if qos not in (0, 1, 2):
            raise ValueError("qos must be 0, 1 or 2")
        if max_retries < 0 or max_retries > 5:
            raise ValueError("max_retries must be between 0 and 5")
        if retry_delay_s < 0 or retry_delay_s > 30:
            raise ValueError("retry_delay_s must be between 0 and 30")
        self._publisher = publisher
        self.device_id = device_id
        self.qos = qos
        self.max_retries = max_retries
        self.retry_delay_s = retry_delay_s
        self.topic = f"face/{device_id}/actions/v1"

    def send(self, envelope: ActionEnvelope):
        validate_action_envelope(envelope)
        payload = json.dumps(envelope.to_dict(), sort_keys=True, separators=(",", ":"))
        attempts = self.max_retries + 1
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                result = self._publisher.publish(self.topic, payload, qos=self.qos, retain=False)
                return getattr(result, "mid", envelope.event_id)
            except (TimeoutError, ConnectionError) as exc:
                last_error = exc
                if attempt + 1 < attempts and self.retry_delay_s:
                    time.sleep(self.retry_delay_s * (2**attempt))
        raise RuntimeError("MQTT action delivery failed") from last_error
