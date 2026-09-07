"""
Unit tests for WebSocket event stream and Edge event publishing.
"""

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from src.api.main import app
from src.api.dependencies import db
from src.api.websocket_manager import ws_manager

client = TestClient(app)


def setup_function() -> None:
    db.use_memory = True
    db._mem_persons.clear()
    db._mem_embeddings.clear()
    db._mem_logs.clear()
    ws_manager.active_connections.clear()


def test_websocket_connect_and_ping_pong() -> None:
    """Verify WebSocket client can connect, receive greeting, and exchange ping/pong."""
    with client.websocket_connect("/ws/v1/events") as websocket:
        greeting = websocket.receive_json()
        assert greeting["event_type"] == "CONNECTED"
        assert "Subscribed" in greeting["message"]

        # Send heartbeat ping
        websocket.send_text("ping")
        pong = websocket.receive_json()
        assert pong["event_type"] == "PONG"


def test_publish_event_broadcasts_to_websocket() -> None:
    """Verify POST /api/v1/events logs access and broadcasts to connected WebSocket clients."""
    with client.websocket_connect("/ws/v1/events") as websocket:
        # Drain initial greeting
        _ = websocket.receive_json()

        # Edge device publishes access event
        payload = {
            "event_type": "ACCESS_EVENT",
            "person_id": "emp_101",
            "name": "Nguyen Van A",
            "device_id": "gate_cam_01",
            "similarity": 0.88,
            "margin": 0.18,
            "status": "MATCHED",
            "liveness": "PASS",
        }
        res = client.post("/api/v1/events", json=payload)
        assert res.status_code == 201
        assert res.json()["status"] == "ok"
        assert res.json()["broadcasted_to"] == 1

        # Client receives broadcast
        received = websocket.receive_json()
        assert received["event_type"] == "ACCESS_EVENT"
        assert received["person_id"] == "emp_101"
        assert received["name"] == "Nguyen Van A"
        assert received["device_id"] == "gate_cam_01"
        assert received["similarity"] == 0.88
        assert received["status"] == "MATCHED"


def test_websocket_api_key_protection(monkeypatch: pytest.MonkeyPatch) -> None:
    """When API_KEY is set, WebSocket must reject unauthorized connections."""
    monkeypatch.setenv("API_KEY", "super-secret-key")

    # Connecting without api_key parameter should be rejected
    with pytest.raises(WebSocketDisconnect) as exc_info:
        with client.websocket_connect("/ws/v1/events"):
            pass
    assert exc_info.value.code == 1008

    # Connecting with invalid api_key should be rejected
    with pytest.raises(WebSocketDisconnect) as exc_info:
        with client.websocket_connect("/ws/v1/events?api_key=wrong_key"):
            pass
    assert exc_info.value.code == 1008

    # Connecting with valid api_key succeeds
    with client.websocket_connect("/ws/v1/events?api_key=super-secret-key") as websocket:
        greeting = websocket.receive_json()
        assert greeting["event_type"] == "CONNECTED"


def test_connection_manager_cleanup_on_disconnect() -> None:
    """Verify active connections list is cleanly decremented upon disconnect."""
    assert len(ws_manager.active_connections) == 0

    with client.websocket_connect("/ws/v1/events") as ws:
        _ = ws.receive_json()
        assert len(ws_manager.active_connections) == 1

    # After exiting context manager, client is disconnected
    assert len(ws_manager.active_connections) == 0
