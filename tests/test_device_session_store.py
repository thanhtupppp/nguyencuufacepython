import pytest

from src.devices.session_store import PostgresDeviceSessionStore


def test_postgres_session_store_has_shared_contract() -> None:
    store = PostgresDeviceSessionStore.__new__(PostgresDeviceSessionStore)
    # Schema/SQL contract is intentionally centralized in the repository class.
    assert hasattr(store, "ensure_schema")
    assert hasattr(store, "upsert")
    assert hasattr(store, "get")


def test_missing_psycopg_fails_explicitly(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.devices.session_store as module

    monkeypatch.setattr(module, "psycopg", None)
    with pytest.raises(RuntimeError, match="psycopg is required"):
        PostgresDeviceSessionStore("dbname=test")
