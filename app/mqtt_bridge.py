"""Bridge canonical RecognitionEvent into the existing device MQTT transport.

The bridge owns no biometric decision logic. It only validates the already
confirmed event and publishes a compact MQTT envelope through MQTTAdapter.
"""
from __future__ import annotations

from typing import Any

from app.fastapi_contract import RecognitionEvent
from src.devices.mqtt_adapter import MQTTAdapter
from src.devices.mqtt_contract import encode


class RecognitionEventMQTTBridge:
    """Publish a canonical RecognitionEvent without creating a second schema."""

    def __init__(self, adapter: MQTTAdapter) -> None:
        self._adapter = adapter

    def publish(self, event: RecognitionEvent):
        payload: dict[str, Any] = {
            "schema_version": 1,
            "event_id": str(event.event_id),
            "event_type": event.event_type.value,
            "camera_id": event.camera_id,
            "device_id": event.device_id,
            "track_id": event.track_id,
            "person_id": event.person_id,
            "model_version": event.model_version,
            "embedding_version": event.embedding_version,
            "quality": event.quality.model_dump(mode="json"),
            "liveness": event.liveness.model_dump(mode="json"),
            "similarity": event.similarity,
            "margin": event.margin,
            "frames_confirmed": event.frames_confirmed,
            "timestamp": event.timestamp.isoformat(),
        }
        return self._adapter.client.publish(
            self._adapter.event_topic,
            payload=encode(payload),
            qos=1,
            retain=False,
        )
