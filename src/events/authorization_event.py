"""Deterministic authorization event identity for transport/action layers.

The event ID is derived only from stable event coordinates. Consumers can use
it as an idempotency key so MQTT/HTTP/WebSocket retries cannot execute the same
physical action twice.
"""
from __future__ import annotations

import hashlib


def authorization_event_id(
    *,
    camera_id: str,
    frame_index: int,
    track_id: str,
    person_id: str,
) -> str:
    if frame_index < 0:
        raise ValueError("frame_index must be >= 0")
    if not camera_id or not track_id or not person_id:
        raise ValueError("camera_id, track_id and person_id are required")
    payload = f"{camera_id}\n{frame_index}\n{track_id}\n{person_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
