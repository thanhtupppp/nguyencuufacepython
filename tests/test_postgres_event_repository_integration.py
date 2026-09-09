import os
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import psycopg
import pytest

from app.event_repository import PostgresEventRepository


pytestmark = pytest.mark.integration


def _dsn() -> str:
    value = os.getenv("DATABASE_URL")
    if not value:
        pytest.skip("DATABASE_URL is required for PostgreSQL integration tests")
    return value


def _ensure_schema(dsn: str) -> None:
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS recognition_events (
                    event_id UUID PRIMARY KEY,
                    payload JSONB NOT NULL,
                    received_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )


def test_restart_persistence_and_no_overwrite() -> None:
    dsn = _dsn()
    _ensure_schema(dsn)
    event_id = uuid4()
    first = {"event_id": str(event_id), "person_id": "person-1", "similarity": 0.91}
    changed = {"event_id": str(event_id), "person_id": "person-2", "similarity": 0.99}

    accepted, stored = PostgresEventRepository(dsn).put_if_absent(event_id, first)
    assert accepted is True
    assert stored == first

    accepted, stored = PostgresEventRepository(dsn).put_if_absent(event_id, changed)
    assert accepted is False
    assert stored == first


def test_concurrent_duplicate_inserts_converge_to_one_payload() -> None:
    dsn = _dsn()
    _ensure_schema(dsn)
    event_id = uuid4()
    payload = {"event_id": str(event_id), "person_id": "person-concurrent", "similarity": 0.95}

    def insert_once() -> tuple[bool, dict]:
        return PostgresEventRepository(dsn).put_if_absent(event_id, payload)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: insert_once(), range(8)))

    assert sum(accepted for accepted, _ in results) == 1
    assert all(stored == payload for _, stored in results)

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM recognition_events WHERE event_id = %s", (event_id,))
            assert cur.fetchone()[0] == 1
