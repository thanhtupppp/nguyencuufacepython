"""MQTT topic and payload contract for face-recognition edge devices.

Transport is deliberately kept separate from recognition decisions: MQTT carries
commands/state/events, while face embeddings and images remain on the configured
HTTP/WebSocket paths. Device identity is part of every topic so multiple cameras
and ESP32 nodes can share one broker safely.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Literal


QOS_COMMAND = 1
QOS_STATE = 1
QOS_EVENT = 1
RETAIN_STATE = True

CommandName = Literal["recognize", "enroll", "health", "config"]


@dataclass(frozen=True)
class DeviceTopics:
    """Canonical topics for one logical device."""

    device_id: str

    @property
    def command(self) -> str:
        return f"face/v1/devices/{self.device_id}/command"

    @property
    def state(self) -> str:
        return f"face/v1/devices/{self.device_id}/state"

    @property
    def event(self) -> str:
        return f"face/v1/devices/{self.device_id}/event"

    @property
    def availability(self) -> str:
        return f"face/v1/devices/{self.device_id}/availability"


def _require_device_id(device_id: str) -> str:
    value = device_id.strip()
    if not value or "/" in value or "#" in value or "+" in value:
        raise ValueError("device_id must be non-empty and MQTT-topic safe")
    return value


def make_topics(device_id: str) -> DeviceTopics:
    return DeviceTopics(_require_device_id(device_id))


def make_command(
    command: CommandName,
    request_id: str,
    *,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a versioned, idempotency-aware device command."""
    if not request_id.strip():
        raise ValueError("request_id is required")
    return {
        "schema_version": 1,
        "request_id": request_id,
        "command": command,
        "payload": payload or {},
    }


def make_state(
    device_id: str,
    *,
    status: str,
    firmware_version: str,
    model_version: str | None = None,
    camera_id: str | None = None,
) -> dict[str, Any]:
    """Build retained device state without leaking embeddings or image data."""
    result: dict[str, Any] = {
        "schema_version": 1,
        "device_id": _require_device_id(device_id),
        "status": status,
        "firmware_version": firmware_version,
    }
    if model_version is not None:
        result["model_version"] = model_version
    if camera_id is not None:
        result["camera_id"] = camera_id
    return result


def make_event(*, request_id: str, event_type: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build a compact event envelope; callers must not put biometrics in payload."""
    if not request_id.strip():
        raise ValueError("request_id is required")
    if not event_type.strip():
        raise ValueError("event_type is required")
    return {
        "schema_version": 1,
        "request_id": request_id,
        "event_type": event_type,
        "payload": payload or {},
    }


def encode(payload: dict[str, Any]) -> bytes:
    """Serialize a contract payload deterministically for MQTT publishing."""
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")


def decode(raw: bytes | str) -> dict[str, Any]:
    """Decode JSON and enforce the presence of the schema version."""
    try:
        value = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid MQTT JSON payload") from exc
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("unsupported or missing schema_version")
    return value
