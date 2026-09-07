"""Device session identity primitives for multi-camera edge connections.

Sessions are transport/runtime identities only. They never contain biometric data.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from threading import RLock
from typing import Optional
from uuid import uuid4


@dataclass(frozen=True)
class DeviceSession:
    session_id: str
    device_id: str
    started_at: datetime
    last_seen_at: datetime
    status: str = "ONLINE"
    sequence: int = 0


class DeviceSessionRegistry:
    """Thread-safe runtime registry; persistent sessions can later move to PostgreSQL."""

    def __init__(self) -> None:
        self._sessions: dict[str, DeviceSession] = {}
        self._lock = RLock()

    def start(self, device_id: str, session_id: Optional[str] = None) -> DeviceSession:
        if not device_id or "/" in device_id or "+" in device_id or "#" in device_id:
            raise ValueError("invalid device_id")
        now = datetime.now(timezone.utc)
        session = DeviceSession(
            session_id=session_id or uuid4().hex,
            device_id=device_id,
            started_at=now,
            last_seen_at=now,
        )
        with self._lock:
            self._sessions[session.session_id] = session
        return session

    def touch(self, session_id: str, device_id: str) -> DeviceSession:
        with self._lock:
            current = self._sessions.get(session_id)
            if current is None:
                raise KeyError("unknown session_id")
            if current.device_id != device_id:
                raise ValueError("session does not belong to device")
            updated = DeviceSession(
                session_id=current.session_id,
                device_id=current.device_id,
                started_at=current.started_at,
                last_seen_at=datetime.now(timezone.utc),
                status="ONLINE",
                sequence=current.sequence + 1,
            )
            self._sessions[session_id] = updated
            return updated

    def close(self, session_id: str, device_id: str) -> DeviceSession:
        with self._lock:
            current = self._sessions.get(session_id)
            if current is None:
                raise KeyError("unknown session_id")
            if current.device_id != device_id:
                raise ValueError("session does not belong to device")
            updated = DeviceSession(
                session_id=current.session_id,
                device_id=current.device_id,
                started_at=current.started_at,
                last_seen_at=datetime.now(timezone.utc),
                status="OFFLINE",
                sequence=current.sequence,
            )
            self._sessions[session_id] = updated
            return updated

    def get(self, session_id: str) -> Optional[DeviceSession]:
        with self._lock:
            return self._sessions.get(session_id)

    def clear(self) -> None:
        with self._lock:
            self._sessions.clear()
