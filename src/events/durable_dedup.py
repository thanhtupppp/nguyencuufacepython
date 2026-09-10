"""Durable event-id ledger for device-side action deduplication.

The ledger records lifecycle state, but cannot make an arbitrary physical side
 effect exactly-once across a process crash. The actuator must therefore make
its operation idempotent using event_id as the command key.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DedupReceipt:
    event_id: str
    device_id: str
    status: str
    duplicate: bool


class SQLiteActionLedger:
    """Durable event ledger suitable for a single device/service process."""

    def __init__(self, path: str | Path) -> None:
        self._path = str(path)
        self._conn = sqlite3.connect(self._path)
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS action_events (
                event_id TEXT PRIMARY KEY,
                device_id TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('CLAIMED','COMPLETED')),
                claimed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                completed_at TEXT
            )"""
        )
        self._conn.commit()

    def claim(self, event_id: str, device_id: str) -> DedupReceipt:
        if not event_id or not device_id:
            raise ValueError("event_id and device_id are required")
        try:
            self._conn.execute(
                "INSERT INTO action_events(event_id, device_id, status) VALUES (?, ?, 'CLAIMED')",
                (event_id, device_id),
            )
            self._conn.commit()
            return DedupReceipt(event_id, device_id, "CLAIMED", False)
        except sqlite3.IntegrityError:
            row = self._conn.execute(
                "SELECT device_id, status FROM action_events WHERE event_id = ?",
                (event_id,),
            ).fetchone()
            if row is None:
                raise RuntimeError("deduplication race lost without a ledger row")
            return DedupReceipt(event_id, row[0], row[1], True)

    def complete(self, event_id: str) -> None:
        if not event_id:
            raise ValueError("event_id is required")
        cur = self._conn.execute(
            "UPDATE action_events SET status='COMPLETED', completed_at=CURRENT_TIMESTAMP "
            "WHERE event_id=? AND status='CLAIMED'",
            (event_id,),
        )
        self._conn.commit()
        if cur.rowcount != 1:
            raise ValueError("event_id is not in CLAIMED state")

    def close(self) -> None:
        self._conn.close()
