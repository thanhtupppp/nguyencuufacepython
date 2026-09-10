from fastapi.testclient import TestClient

from src.api.app import create_app


def test_healthz() -> None:
    client = TestClient(create_app())
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readyz_false_by_default() -> None:
    client = TestClient(create_app())
    response = client.get("/readyz")
    assert response.status_code == 200
    assert response.json() == {"ready": False}


def test_readyz_can_be_enabled() -> None:
    client = TestClient(create_app(engine_ready=True))
    response = client.get("/readyz")
    assert response.status_code == 200
    assert response.json() == {"ready": True}
