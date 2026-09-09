from datetime import datetime, timezone
from uuid import uuid4

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def payload(**overrides):
    data = {
        "event_id": str(uuid4()),
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
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    data.update(overrides)
    return data


def test_healthz():
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_http_accepts_valid_event_and_is_idempotent():
    data = payload()
    first = client.post("/v1/events", json=data)
    second = client.post("/v1/events", json=data)

    assert first.status_code == 202
    assert first.json()["status"] == "accepted"
    assert second.status_code == 200
    assert second.json()["status"] == "duplicate"
    assert second.json()["event"] == first.json()["event"]


def test_http_rejects_unconfirmed_event_fields():
    response = client.post("/v1/events", json=payload(person_id=None))
    assert response.status_code == 422


def test_websocket_serializes_same_canonical_payload():
    data = payload()
    with client.websocket_connect("/v1/events/ws") as websocket:
        websocket.send_json(data)
        response = websocket.receive_json()
        assert response["status"] == "accepted"
        assert response["event"]["event_id"] == data["event_id"]
        assert response["event"]["timestamp"].endswith("+00:00")
