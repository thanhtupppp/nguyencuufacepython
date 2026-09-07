"""Idempotency stores for MQTT command delivery.

The protocol is at-least-once, so command execution must be deduplicated outside
an individual MQTT client process when the service is horizontally scaled.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol


class IdempotencyStore(Protocol):
    def claim(self, device_id: str, request_id: str) -> bool:
        """Atomically claim a request. True means this worker owns execution."""


@dataclass
class _Entry:
    expires_at: float


class TTLMemoryIdempotencyStore:
    """Bounded single-process fallback with TTL.

    This is safe for one worker only. Production multi-worker deployments should
    use PostgresIdempotencyStore (or another shared atomic store).
    """

    def __init__(self, ttl_seconds: float = 300.0, max_entries: int = 100_000) -> None:
        if ttl_seconds <= 0 or max_entries <= 0:
            raise ValueError("ttl_seconds and max_entries must be positive")
        self.ttl_seconds = float(ttl_seconds)
        self.max_entries = int(max_entries)
        self._entries: dict[tuple[str, str], _Entry] = {}
        self._lock = threading.Lock()

    def claim(self, device_id: str, request_id: str) -> bool:
        key = (device_id, request_id)
        now = time.monotonic()
        with self._lock:
            self._purge(now)
            if key in self._entries:
                return False
            if len(self._entries) >= self.max_entries:
                oldest = min(self._entries, key=lambda k: self._entries[k].expires_at)
                del self._entries[oldest]
            self._entries[key] = _Entry(now + self.ttl_seconds)
            return True

    def _purge(self, now: float) -> None:
        expired = [key for key, entry in self._entries.items() if entry.expires_at <= now]
        for key in expired:
            del self._entries[key]


class PostgresIdempotencyStore:
    """Shared atomic idempotency store backed by PostgreSQL.

    ``connection_factory`` must return a psycopg connection. Claiming uses a
    primary-key insert, so two service workers cannot both win the same request.
    Expired rows are opportunistically deleted before the insert.
    """

    def __init__(self, connection_factory: Callable[[], Any], ttl_seconds: int = 300) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        self.connection_factory = connection_factory
        self.ttl_seconds = int(ttl_seconds)

    def claim(self, device_id: str, request_id: str) -> bool:
        conn = self.connection_factory()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM mqtt_idempotency WHERE expires_at <= CURRENT_TIMESTAMP"
                )
                cur.execute(
                    """
                    INSERT INTO mqtt_idempotency (device_id, request_id, expires_at)
                    VALUES (%s, %s, CURRENT_TIMESTAMP + (%s * INTERVAL '1 second'))
                    ON CONFLICT (device_id, request_id) DO NOTHING
                    RETURNING device_id
                    """,
                    (device_id, request_id, self.ttl_seconds),
                )
                claimed = cur.fetchone() is not None
            conn.commit()
            return claimed
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
