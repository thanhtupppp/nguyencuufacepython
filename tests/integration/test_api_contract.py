import numpy as np
from fastapi.testclient import TestClient

from src.api import routes
from src.api.routes import faces
from src.api.main import create_app


class FakePipeline:
    def extract_best_face(self, image):
        class Result:
            embedding = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            model_version = "arcface_v1"
            quality_score = 0.95

        return Result()


class FakeDB:
    def __init__(self):
        self.persons = {"P001": {"person_id": "P001", "name": "Test Person"}}
        self.embeddings = []

    def get_person(self, person_id):
        return self.persons.get(person_id)

    def add_embedding(self, person_id, embedding, model_version, quality, source):
        self.embeddings.append((person_id, model_version, float(quality)))
        return 1

    def recognize_with_margin(self, embedding, model_version, threshold, margin):
        class Candidate:
            person_id = "P001"
            similarity = 0.91

            def __init__(self):
                self.__dict__ = {"person_id": self.person_id, "similarity": self.similarity}

        class Decision:
            status = "RECOGNIZED"
            person_id = "P001"
            similarity = 0.91
            second_similarity = 0.20
            margin = 0.71
            top_candidates = [Candidate()]

        return Decision()

    def log_access(self, *args):
        return None

    def search_top_k(self, embedding, model_version, top_k):
        class Candidate:
            person_id = "P001"
            similarity = 0.91

        return [Candidate()]


def _jpeg_bytes():
    import cv2

    image = np.full((16, 16, 3), 127, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded.tobytes()


def _client(monkeypatch, pipeline=True):
    fake_db = FakeDB()
    monkeypatch.setattr(faces, "db", fake_db)
    monkeypatch.setattr(faces, "recognition_pipeline", FakePipeline() if pipeline else None)
    return TestClient(create_app())


def test_enroll_contract(monkeypatch):
    client = _client(monkeypatch)
    response = client.post(
        "/api/v1/faces/enroll",
        data={"person_id": "P001", "quality_score": "0.9"},
        files={"image": ("face.jpg", _jpeg_bytes(), "image/jpeg")},
    )
    assert response.status_code == 201
    body = response.json()
    assert body["person_id"] == "P001"
    assert body["model_version"] == "arcface_v1"


def test_recognize_contract_preserves_unknown_safe_fields(monkeypatch):
    client = _client(monkeypatch)
    response = client.post(
        "/api/v1/faces/recognize",
        data={"model_version": "arcface_v1", "threshold": "0.60", "margin": "0.08"},
        files={"image": ("face.jpg", _jpeg_bytes(), "image/jpeg")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "RECOGNIZED"
    assert body["person_id"] == "P001"
    assert body["model_version"] == "arcface_v1"
    assert "embedding" not in body


def test_model_version_mismatch_is_rejected(monkeypatch):
    client = _client(monkeypatch)
    response = client.post(
        "/api/v1/faces/recognize",
        data={"model_version": "adaface_v1"},
        files={"image": ("face.jpg", _jpeg_bytes(), "image/jpeg")},
    )
    assert response.status_code == 409


def test_recognition_fails_closed_without_pipeline(monkeypatch):
    client = _client(monkeypatch, pipeline=False)
    response = client.post(
        "/api/v1/faces/recognize",
        files={"image": ("face.jpg", _jpeg_bytes(), "image/jpeg")},
    )
    assert response.status_code == 503
