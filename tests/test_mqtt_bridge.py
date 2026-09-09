from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import Mock
from uuid import uuid4

from app.fastapi_contract import RecognitionEvent
from app.mqtt_bridge import RecognitionEventMQTTBridge


def sample_event() -> RecognitionEvent:
    return RecognitionEvent(
        event_id=uuid4(),
        camera_id="cam-01",
        device_id="pi-01",
        track_id="track-7",
        state="RECOGNITION_CONFIRMED",
        person_id="person-42",
        model_version="scrfd-arcface-v1",
        embedding_version="arcface-v1",
        quality=0.93,
        liveness=0.99,
        similarity=0.88,
        margin=0.14,
        frames_confirmed=4,
        timestamp=datetime.now(timezone.utc),
    )


def test_bridge_publishes_canonical_fields_without_embedding() -> None:
    adapter = Mock()
    adapter.event_topic = "face/v1/devices/pi-01/event"
    adapter.client.publish.return_value = Mock()

    result = RecognitionEventMQTTBridge(adapter).publish(sample_event())

    assert result is adapter.client.publish.return_value
    adapter.client.publish.assert_called_once()
    _, kwargs = adapter.client.publish.call_args
    assert kwargs["qos"] == 1
    assert kwargs["retain"] is False
    assert b"person-42" in kwargs["payload"]
    assert b"embedding" not in kwargs["payload"].lower()
