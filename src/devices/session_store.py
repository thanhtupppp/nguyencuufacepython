"""PostgreSQL persistence for device sessions used by multi-worker deployments."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

try:
    import psycopg  # type: ignore
except ImportError:  # pragma: no cover
    psycopg = None


class PostgresDeviceSessionStore:
    """Small repository around the device_sessions table.

    Session writes are intentionally separate from biometric tables so device
    connection churn cannot affect face embeddings or recognition decisions.
    """

    def __init__(self, conninfo: str) -> None:
        if psycopg is None:
            raise RuntimeError("psycopg is required for PostgreSQL device sessions")
        self.conninfo = conninfo

    def ensure_schema(self) -> None:
        with psycopg.connect(self.conninfo) as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS device_sessions (
                    session_id VARCHAR(64) PRIMARY KEY,
                    device_id VARCHAR(64) NOT NULL REFERENCES devices(device_id) ON DELETE CASCADE,
                    started_at TIMESTAMPTZ NOT NULL,
                    last_seen_at TIMESTAMPTZ NOT NULL,
                    status VARCHAR(16) NOT NULL,
                    sequence BIGINT NOT NULL DEFAULT 0
                )"""
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_device_sessions_device ON device_sessions(device_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_device_sessions_last_seen ON device_sessions(last_seen_at)"
            )

    def upsert(self, session_id: str, device_id: str, status: str, sequence: int, *, started_at: Optional[datetime] = None) -> None:
        now = datetime.now(timezone.utc)
        started = started_at or now
        with psycopg.connect(self.conninfo) as conn:
            conn.execute(
                """INSERT INTO device_sessions
                   (session_id, device_id, started_at, last_seen_at, status, sequence)
                   VALUES (%s, %s, %s, %s, %s, %s)
                   ON CONFLICT (session_id) DO UPDATE SET
                     last_seen_at = EXCLUDED.last_seen_at,
                     status = EXCLUDED.status,
                     sequence = EXCLUDED.sequence
                """,
                (session_id, device_id, started, now, status, sequence),
            )

    def get(self, session_id: str) -> Optional[dict[str, Any]]:
        with psycopg.connect(self.conninfo) as conn:
            row = conn.execute(
                "SELECT session_id, device_id, started_at, last_seen_at, status, sequence FROM device_sessions WHERE session_id=%s",
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "session_id": row[0],
            "device_id": row[1],
            "started_at": row[2],
            "last_seen_at": row[3],
            "status": row[4],
            "sequence": row[5],
        }
