"""Durable RecognitionEvent repository for PostgreSQL.

The repository persists canonical transport events only; it never performs
biometric inference or creates person_id values.
"""
from __future__ import annotations

from typing import Any, Protocol
from uuid import UUID

import psycopg
from psycopg.types.json import Jsonb


class EventRepository(Protocol):
    def put_if_absent(self, event_id: UUID, payload: dict[str, Any]) -> tuple[bool, dict[str, Any]]: ...


class InMemoryEventRepository:
    """Deterministic test double; never used as the production default."""

    def __init__(self) -> None:
        self._events: dict[UUID, dict[str, Any]] = {}

    def put_if_absent(self, event_id: UUID, payload: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        if event_id in self._events:
            return False, self._events[event_id]
        self._events[event_id] = dict(payload)
        return True, self._events[event_id]


class PostgresEventRepository:
    def __init__(self, dsn: str):
        self.dsn = dsn

    def put_if_absent(self, event_id: UUID, payload: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        """Atomically insert or return the original payload for event_id."""
        with psycopg.connect(self.dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO recognition_events (event_id, payload)
                    VALUES (%s, %s)
                    ON CONFLICT (event_id) DO NOTHING
                    RETURNING payload
                    """,
                    (event_id, Jsonb(payload)),
                )
                row = cur.fetchone()
                if row is not None:
                    return True, dict(row[0])

                cur.execute(
                    "SELECT payload FROM recognition_events WHERE event_id = %s",
                    (event_id,),
                )
                row = cur.fetchone()
                if row is None:
                    raise RuntimeError("event disappeared after conflict")
                return False, dict(row[0])
