"""Bounded MQTT action transports with sync and async retry semantics."""
from __future__ import annotations

import asyncio
import json
from typing import Protocol

from .action_envelope import ActionEnvelope
from .transport import validate_action_envelope


class MqttPublisher(Protocol):
    def publish(self, topic: str, payload: str, qos: int, retain: bool): ...


class MqttActionTransport:
    """Synchronous adapter; retry delays block the calling thread only."""

    TRANSIENT_EXCEPTIONS = (TimeoutError, ConnectionError)

    def __init__(self, publisher: MqttPublisher, device_id: str, qos: int = 1, *, max_retries: int = 2, retry_delay_s: float = 0.05, max_backoff_s: float = 2.0):
        if not device_id or "/" in device_id or "#" in device_id or "+" in device_id:
            raise ValueError("device_id is required and MQTT-topic safe")
        if qos not in (0, 1, 2):
            raise ValueError("qos must be 0, 1 or 2")
        if max_retries < 0 or max_retries > 5:
            raise ValueError("max_retries must be between 0 and 5")
        if not 0 <= retry_delay_s <= 30 or not 0 < max_backoff_s <= 30:
            raise ValueError("invalid retry delay bounds")
        self._publisher = publisher
        self.device_id = device_id
        self.qos = qos
        self.max_retries = max_retries
        self.retry_delay_s = retry_delay_s
        self.max_backoff_s = max_backoff_s
        self.topic = f"face/{device_id}/actions/v1"

    def send(self, envelope: ActionEnvelope):
        validate_action_envelope(envelope)
        payload = json.dumps(envelope.to_dict(), sort_keys=True, separators=(",", ":"))
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                result = self._publisher.publish(self.topic, payload, qos=self.qos, retain=False)
                return getattr(result, "mid", envelope.event_id)
            except self.TRANSIENT_EXCEPTIONS as exc:
                last_error = exc
                if attempt < self.max_retries:
                    import time
                    time.sleep(min(self.retry_delay_s * (2**attempt), self.max_backoff_s))
        raise RuntimeError("MQTT action delivery failed") from last_error


class AsyncMqttPublisher(Protocol):
    async def publish(self, topic: str, payload: str, qos: int, retain: bool): ...


class AsyncMqttActionTransport:
    """Async adapter that never blocks the event loop during retry backoff."""

    TRANSIENT_EXCEPTIONS = MqttActionTransport.TRANSIENT_EXCEPTIONS

    def __init__(self, publisher: AsyncMqttPublisher, device_id: str, qos: int = 1, *, max_retries: int = 2, retry_delay_s: float = 0.05, max_backoff_s: float = 2.0):
        if not device_id or "/" in device_id or "#" in device_id or "+" in device_id:
            raise ValueError("device_id is required and MQTT-topic safe")
        if qos not in (0, 1, 2) or max_retries < 0 or max_retries > 5:
            raise ValueError("invalid MQTT retry configuration")
        self._publisher = publisher
        self.topic = f"face/{device_id}/actions/v1"
        self.qos, self.max_retries = qos, max_retries
        self.retry_delay_s, self.max_backoff_s = retry_delay_s, max_backoff_s

    async def send(self, envelope: ActionEnvelope):
        validate_action_envelope(envelope)
        payload = json.dumps(envelope.to_dict(), sort_keys=True, separators=(",", ":"))
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                result = await self._publisher.publish(self.topic, payload, qos=self.qos, retain=False)
                return getattr(result, "mid", envelope.event_id)
            except self.TRANSIENT_EXCEPTIONS as exc:
                last_error = exc
                if attempt < self.max_retries:
                    await asyncio.sleep(min(self.retry_delay_s * (2**attempt), self.max_backoff_s))
        raise RuntimeError("MQTT action delivery failed") from last_error
