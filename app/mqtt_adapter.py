"""MQTT transport for the canonical RecognitionEvent.

This module deliberately contains no biometric decision logic. It publishes an
already validated RecognitionEvent using the same JSON representation as the
HTTP/WebSocket boundary.
"""
from __future__ import annotations

import json
from typing import Callable

import paho.mqtt.client as mqtt

from app.fastapi_contract import RecognitionEvent


class MqttPublisher:
    def __init__(self, client: mqtt.Client, topic_prefix: str = "face/events/v1") -> None:
        self._client = client
        self._topic_prefix = topic_prefix.rstrip("/")

    def publish(self, event: RecognitionEvent) -> mqtt.MQTTMessageInfo:
        topic = f"{self._topic_prefix}/{event.camera_id}"
        payload = event.model_dump_json()
        return self._client.publish(topic, payload, qos=1, retain=False)


def build_client(
    client_id: str,
    on_message: Callable | None = None,
) -> mqtt.Client:
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
    if on_message is not None:
        client.on_message = on_message
    return client


def decode_event(payload: bytes | str) -> RecognitionEvent:
    """Decode only the canonical event; invalid payloads fail closed."""
    raw = payload.decode("utf-8") if isinstance(payload, bytes) else payload
    return RecognitionEvent.model_validate(json.loads(raw))
