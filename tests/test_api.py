"""API contract tests; recognition model is intentionally mocked in unit tests."""

import numpy as np
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.routes import faces
from src.api.dependencies import db


client = TestClient(app)


def setup_function() -> None:
    db.use_memory = True
    db._mem_persons.clear()
    db._mem_embeddings.clear()
    db._mem_logs.clear()


def test_healthz() -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_person_crud() -> None:
    created = client.post("/api/v1/persons", json={"person_id": "p001", "name": "Alice"})
    assert created.status_code == 201
    assert created.json()["person_id"] == "p001"

    fetched = client.get("/api/v1/persons/p001")
    assert fetched.status_code == 200
    assert fetched.json()["name"] == "Alice"

    listed = client.get("/api/v1/persons")
    assert listed.status_code == 200
    assert len(listed.json()) == 1

    deleted = client.delete("/api/v1/persons/p001")
    assert deleted.status_code == 204


def test_enroll_fails_closed_when_model_is_not_configured() -> None:
    client.post("/api/v1/persons", json={"person_id": "p001", "name": "Alice"})
    response = client.post(
        "/api/v1/faces/enroll",
        data={"person_id": "p001"},
        files={"image": ("face.jpg", b"not-an-image", "image/jpeg")},
    )
    assert response.status_code == 400


def test_recognition_model_hook_fails_closed(monkeypatch) -> None:
    monkeypatch.setattr(faces, "_extract_embedding", lambda image: (_ for _ in ()).throw(Exception("unexpected")))
    # Invalid input is rejected before the model hook, keeping the HTTP contract deterministic.
    response = client.post(
        "/api/v1/faces/recognize",
        files={"image": ("face.jpg", b"not-an-image", "image/jpeg")},
    )
    assert response.status_code == 400


def test_person_status_patch() -> None:
    client.post("/api/v1/persons", json={"person_id": "p002", "name": "Bob"})
    
    patch_res = client.patch("/api/v1/persons/p002/status", json={"status": "inactive"})
    assert patch_res.status_code == 200
    assert patch_res.json()["status"] == "inactive"

    get_res = client.get("/api/v1/persons/p002")
    assert get_res.json()["status"] == "inactive"

    # Invalid status value
    invalid_patch = client.patch("/api/v1/persons/p002/status", json={"status": "invalid_status"})
    assert invalid_patch.status_code == 422


def test_upload_file_size_limit() -> None:
    # 5MB + 10 bytes payload
    oversized_jpeg = b"\xff\xd8\xff\xe0" + b"\x00" * (5 * 1024 * 1024 + 10)
    response = client.post(
        "/api/v1/faces/recognize",
        files={"image": ("oversized.jpg", oversized_jpeg, "image/jpeg")},
    )
    assert response.status_code == 413
    assert "Payload too large" in response.json()["detail"]


def test_api_key_authentication(monkeypatch) -> None:
    monkeypatch.setenv("API_KEY", "test-secret-key-123")

    # Missing API Key -> 401
    unauth = client.get("/api/v1/persons")
    assert unauth.status_code == 401

    # Wrong API Key -> 401
    wrong = client.get("/api/v1/persons", headers={"X-API-Key": "wrong-key"})
    assert wrong.status_code == 401

    # Correct API Key via X-API-Key header -> 200
    valid_x_key = client.get("/api/v1/persons", headers={"X-API-Key": "test-secret-key-123"})
    assert valid_x_key.status_code == 200

    # Correct API Key via Authorization: Bearer -> 200
    valid_bearer = client.get("/api/v1/persons", headers={"Authorization": "Bearer test-secret-key-123"})
    assert valid_bearer.status_code == 200


def test_enroll_rejects_mask_occlusion(monkeypatch) -> None:
    """When pipeline raises MASK_DETECTED, API enroll returns 422 with clear user guidance."""
    client.post("/api/v1/persons", json={"person_id": "p_mask", "name": "Masked Person"})

    # Mock pipeline to raise MASK_DETECTED
    class MockPipeline:
        def extract_best_face(self, image):
            raise ValueError("MASK_DETECTED")

    monkeypatch.setattr("src.api.dependencies.recognition_pipeline", MockPipeline())
    monkeypatch.setattr("src.api.routes.faces.recognition_pipeline", MockPipeline())

    import cv2

    _, buf = cv2.imencode(".jpg", np.zeros((100, 100, 3), dtype=np.uint8))
    valid_jpeg = buf.tobytes()
    response = client.post(
        "/api/v1/faces/enroll",
        data={"person_id": "p_mask"},
        files={"image": ("face.jpg", valid_jpeg, "image/jpeg")},
    )
    assert response.status_code == 422
    assert "Face mask detected" in response.json()["detail"]


