from datetime import datetime, timezone
from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.fastapi_contract import RecognitionEvent


def base(**overrides):
    data = {
        "event_id": uuid4(),
        "camera_id": "camera-01",
        "device_id": "edge-01",
        "track_id": "track-01",
        "state": "RECOGNITION_CONFIRMED",
        "person_id": "person-000123",
        "model_version": "arcface:test",
        "embedding_version": "v1",
        "quality": 0.95,
        "liveness": 0.99,
        "similarity": 0.82,
        "margin": 0.11,
        "frames_confirmed": 3,
        "timestamp": datetime.now(timezone.utc),
    }
    data.update(overrides)
    return data


def test_confirmed_event_is_valid():
    event = RecognitionEvent(**base())
    assert event.state == "RECOGNITION_CONFIRMED"
    assert event.timestamp.tzinfo is not None


def test_confirmed_event_requires_person_id():
    with pytest.raises(ValidationError):
        RecognitionEvent(**base(person_id=None))


def test_confirmed_event_requires_positive_confirmation_frames():
    with pytest.raises(ValidationError):
        RecognitionEvent(**base(frames_confirmed=0))


def test_naive_timestamp_is_rejected():
    with pytest.raises(ValidationError):
        RecognitionEvent(**base(timestamp=datetime.now()))
