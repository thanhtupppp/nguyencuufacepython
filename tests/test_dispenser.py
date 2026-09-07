"""
Tests for Smart Facial Recognition Toilet Paper Dispenser endpoint,
Auto-enrollment, Anti-Abuse 5-minute cooldown, and Dispenser Stats.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.dependencies import db
import src.api.routes.dispenser as dispenser_module
from src.recognition.pipeline import FaceEmbeddingResult

client = TestClient(app)


def _make_dummy_jpeg() -> bytes:
    """Creates a minimal authentic JPEG image buffer."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


@pytest.fixture(autouse=True)
def clean_db():
    orig_sqlite = db.use_sqlite
    orig_memory = db.use_memory
    db.use_sqlite = False
    db.use_memory = True
    db._mem_persons.clear()
    db._mem_embeddings.clear()
    db._mem_logs.clear()
    db._mem_dispense_logs.clear()
    yield
    db.use_sqlite = orig_sqlite
    db.use_memory = orig_memory


def test_dispenser_no_face_detected(monkeypatch):
    mock_pipe = MagicMock()
    mock_pipe.extract_best_face.side_effect = ValueError("NO_FACE_DETECTED")
    monkeypatch.setattr(dispenser_module, "recognition_pipeline", mock_pipe)

    jpeg = _make_dummy_jpeg()
    response = client.post(
        "/api/v1/dispenser/request-paper",
        files={"image": ("camera.jpg", jpeg, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["granted"] is False
    assert data["status"] == "NO_FACE_DETECTED"
    assert "Không tìm thấy khuôn mặt" in data["message"]


def test_dispenser_mask_detected(monkeypatch):
    mock_pipe = MagicMock()
    mock_pipe.extract_best_face.side_effect = ValueError("MASK_DETECTED")
    monkeypatch.setattr(dispenser_module, "recognition_pipeline", mock_pipe)

    jpeg = _make_dummy_jpeg()
    response = client.post(
        "/api/v1/dispenser/request-paper",
        files={"image": ("camera.jpg", jpeg, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["granted"] is False
    assert data["status"] == "MASK_DETECTED"
    assert "khẩu trang" in data["message"]


def test_dispenser_occlusion_detected(monkeypatch):
    mock_pipe = MagicMock()
    mock_pipe.extract_best_face.side_effect = ValueError("OCCLUSION_DETECTED")
    monkeypatch.setattr(dispenser_module, "recognition_pipeline", mock_pipe)

    jpeg = _make_dummy_jpeg()
    response = client.post(
        "/api/v1/dispenser/request-paper",
        files={"image": ("camera.jpg", jpeg, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["granted"] is False
    assert data["status"] == "OCCLUSION_DETECTED"
    assert "che khuất" in data["message"]


def test_dispenser_new_user_auto_enroll_and_grant(monkeypatch):
    # Setup mock pipeline returning a 512D unit vector
    dummy_vec = np.random.randn(512).astype(np.float32)
    dummy_vec /= np.linalg.norm(dummy_vec)

    mock_result = FaceEmbeddingResult(
        embedding=dummy_vec,
        bbox=[10.0, 10.0, 80.0, 80.0],
        landmarks=np.zeros((5, 2)),
        quality_score=0.95,
        model_version="arcface_v1",
        detector_score=0.99,
    )
    mock_pipe = MagicMock()
    mock_pipe.extract_best_face.return_value = mock_result
    monkeypatch.setattr(dispenser_module, "recognition_pipeline", mock_pipe)

    jpeg = _make_dummy_jpeg()
    response = client.post(
        "/api/v1/dispenser/request-paper",
        files={"image": ("camera.jpg", jpeg, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()

    # Must be granted as new user
    assert data["granted"] is True
    assert data["status"] == "NEW_USER_GRANTED"
    assert data["is_new_user"] is True
    assert data["person_id"].startswith("USER_")
    assert "Chào mừng người dùng mới" in data["message"]
    assert data["pulse_ms"] == 2500

    # Verify user was saved into Database
    created_id = data["person_id"]
    person = db.get_person(created_id)
    assert person is not None
    assert len(db._mem_embeddings) == 1
    assert db._mem_embeddings[0]["person_id"] == created_id


def test_dispenser_cooldown_blocked_and_reallowed(monkeypatch):
    dummy_vec = np.random.randn(512).astype(np.float32)
    dummy_vec /= np.linalg.norm(dummy_vec)

    mock_result = FaceEmbeddingResult(
        embedding=dummy_vec,
        bbox=[10.0, 10.0, 80.0, 80.0],
        landmarks=np.zeros((5, 2)),
        quality_score=0.98,
        model_version="arcface_v1",
        detector_score=0.99,
    )
    mock_pipe = MagicMock()
    mock_pipe.extract_best_face.return_value = mock_result
    monkeypatch.setattr(dispenser_module, "recognition_pipeline", mock_pipe)

    jpeg = _make_dummy_jpeg()

    # 1. First request -> Auto-enrolled and GRANTED
    res1 = client.post(
        "/api/v1/dispenser/request-paper",
        files={"image": ("camera.jpg", jpeg, "image/jpeg")},
        data={"cooldown_minutes": 5.0},
    )
    assert res1.status_code == 200
    d1 = res1.json()
    assert d1["granted"] is True
    person_id = d1["person_id"]

    # 2. Second request immediately (0 seconds later) -> Must be DENIED due to 5-min cooldown
    res2 = client.post(
        "/api/v1/dispenser/request-paper",
        files={"image": ("camera.jpg", jpeg, "image/jpeg")},
        data={"cooldown_minutes": 5.0},
    )
    assert res2.status_code == 200
    d2 = res2.json()
    assert d2["granted"] is False
    assert d2["status"] == "COOLDOWN_BLOCKED"
    assert d2["person_id"] == person_id
    assert d2["cooldown_remaining_seconds"] > 0
    assert "tránh lãng phí giấy vệ sinh" in d2["message"]

    # 3. Fast-forward last dispense timestamp to 301 seconds ago
    past_time = datetime.now(timezone.utc) - timedelta(seconds=301)
    db._mem_dispense_logs[0]["timestamp"] = past_time

    # 4. Third request -> Cooldown passed -> Must be GRANTED again
    res3 = client.post(
        "/api/v1/dispenser/request-paper",
        files={"image": ("camera.jpg", jpeg, "image/jpeg")},
        data={"cooldown_minutes": 5.0},
    )
    assert res3.status_code == 200
    d3 = res3.json()
    assert d3["granted"] is True
    assert d3["status"] == "GRANTED"
    assert d3["is_new_user"] is False
    assert d3["person_id"] == person_id
    assert "Đang cấp giấy vệ sinh" in d3["message"]


def test_dispenser_stats_and_logs(monkeypatch):
    dummy_vec = np.random.randn(512).astype(np.float32)
    dummy_vec /= np.linalg.norm(dummy_vec)

    mock_result = FaceEmbeddingResult(
        embedding=dummy_vec,
        bbox=[10.0, 10.0, 80.0, 80.0],
        landmarks=np.zeros((5, 2)),
        quality_score=0.98,
        model_version="arcface_v1",
        detector_score=0.99,
    )
    mock_pipe = MagicMock()
    mock_pipe.extract_best_face.return_value = mock_result
    monkeypatch.setattr(dispenser_module, "recognition_pipeline", mock_pipe)

    jpeg = _make_dummy_jpeg()

    # 1. Request -> Granted
    client.post("/api/v1/dispenser/request-paper", files={"image": ("camera.jpg", jpeg, "image/jpeg")})
    # 2. Immediate request -> Blocked
    client.post("/api/v1/dispenser/request-paper", files={"image": ("camera.jpg", jpeg, "image/jpeg")})

    # Query Stats
    stats_res = client.get("/api/v1/dispenser/stats")
    assert stats_res.status_code == 200
    stats = stats_res.json()
    assert stats["total_granted"] == 1
    assert stats["total_blocked"] == 1
    assert stats["unique_users"] == 1

    # Query Logs
    logs_res = client.get("/api/v1/dispenser/logs")
    assert logs_res.status_code == 200
    logs = logs_res.json()
    assert len(logs) == 2
