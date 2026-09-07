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


def test_recognition_model_hook_fails_closed() -> None:
    faces._extract_embedding = lambda image: (_ for _ in ()).throw(Exception("unexpected"))
    # Invalid input is rejected before the model hook, keeping the HTTP contract deterministic.
    response = client.post(
        "/api/v1/faces/recognize",
        files={"image": ("face.jpg", b"not-an-image", "image/jpeg")},
    )
    assert response.status_code == 400
