from datetime import datetime, timezone
from uuid import uuid4

from fastapi.testclient import TestClient

from app.event_repository import InMemoryEventRepository
from app.main import create_app
from src.health.runtime import RuntimeLifecycle


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


def ready_app():
    runtime = RuntimeLifecycle()
    probes = {name: (lambda name=name: (True, "test_ready")) for name in runtime.gate.snapshot()}
    return create_app(InMemoryEventRepository(), readiness=runtime, dependency_probes=probes)


def test_healthz_is_live_before_dependencies_are_ready():
    app = create_app(InMemoryEventRepository())
    with TestClient(app) as client:
        response = client.get("/healthz")
        assert response.status_code == 200


def test_readyz_is_fail_closed_then_ready():
    app = create_app(InMemoryEventRepository())
    with TestClient(app) as client:
        response = client.get("/readyz")
        assert response.status_code == 503

    app = ready_app()
    with TestClient(app) as client:
        response = client.get("/readyz")
        assert response.status_code == 200
        assert response.json()["ready"] is True


def test_request_id_is_generated_and_returned():
    with TestClient(create_app(InMemoryEventRepository())) as client:
        response = client.post("/v1/events", json=payload())
        assert response.status_code == 202
        assert response.headers["X-Request-ID"]
        assert response.json()["request_id"] == response.headers["X-Request-ID"]


def test_request_id_is_preserved_when_valid():
    request_id = "req-test-123"
    with TestClient(create_app(InMemoryEventRepository())) as client:
        response = client.post("/v1/events", headers={"X-Request-ID": request_id}, json=payload())
        assert response.headers["X-Request-ID"] == request_id


def test_http_accepts_valid_event_and_is_idempotent():
    with TestClient(create_app(InMemoryEventRepository())) as client:
        data = payload()
        first = client.post("/v1/events", json=data)
        second = client.post("/v1/events", json=data)
        assert first.status_code == 202
        assert first.json()["status"] == "accepted"
        assert second.status_code == 200
        assert second.json()["status"] == "duplicate"
        assert second.json()["event"] == first.json()["event"]


def test_http_rejects_unconfirmed_event_fields():
    with TestClient(create_app(InMemoryEventRepository())) as client:
        response = client.post("/v1/events", json=payload(person_id=None))
        assert response.status_code == 422


def test_websocket_serializes_same_canonical_payload_and_request_id():
    data = payload()
    with TestClient(create_app(InMemoryEventRepository())) as client:
        with client.websocket_connect("/v1/events/ws", headers={"X-Request-ID": "ws-req-1"}) as websocket:
            websocket.send_json(data)
            response = websocket.receive_json()
            assert response["status"] == "accepted"
            assert response["event"]["event_id"] == data["event_id"]
            assert response["request_id"] == "ws-req-1"
