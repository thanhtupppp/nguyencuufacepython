from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from typing import Any, Callable, Optional

import paho.mqtt.client as mqtt

from src.devices.mqtt_protocol import decode, encode, make_command, make_event, make_state


@dataclass(frozen=True)
class MQTTSettings:
    host: str = "127.0.0.1"
    port: int = 1883
    keepalive: int = 60
    client_id: str = ""


class InMemoryIdempotencyStore:
    def __init__(self) -> None:
        self._seen: set[tuple[str, str]] = set()
        self._lock = threading.Lock()

    def claim(self, device_id: str, request_id: str) -> bool:
        key = (device_id, request_id)
        with self._lock:
            if key in self._seen:
                return False
            self._seen.add(key)
            return True


class MQTTAdapter:
    def __init__(
        self,
        settings: MQTTSettings,
        *,
        device_id: str,
        command_handler: Optional[Callable[[dict[str, Any]], None]] = None,
        idempotency_store: Optional[InMemoryIdempotencyStore] = None,
    ) -> None:
        self.settings = settings
        self.device_id = device_id
        self.command_handler = command_handler
        self.idempotency_store = idempotency_store or InMemoryIdempotencyStore()
        self.command_topic = f"devices/{device_id}/command"
        self.state_topic = f"devices/{device_id}/state"
        self.event_topic = f"devices/{device_id}/event"
        self.availability_topic = f"devices/{device_id}/availability"
        self._connected = threading.Event()

        self.client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id=settings.client_id or f"service-{device_id}",
        )
        self.client.will_set(self.availability_topic, payload="offline", qos=1, retain=True)
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message

    def connect(self) -> None:
        self.client.connect(self.settings.host, self.settings.port, self.settings.keepalive)
        self.client.loop_start()

    def disconnect(self) -> None:
        self._connected.clear()
        self.client.loop_stop()
        self.client.disconnect()

    def publish_state(self, *, status: str, firmware_version: str, model_version: str | None = None, camera_id: str | None = None) -> mqtt.MQTTMessageInfo:
        body = make_state(self.device_id, status=status, firmware_version=firmware_version, model_version=model_version, camera_id=camera_id)
        return self.client.publish(self.state_topic, payload=encode(body), qos=1, retain=True)

    def publish_event(self, *, request_id: str, event_type: str, payload: Optional[dict[str, Any]] = None) -> mqtt.MQTTMessageInfo:
        body = make_event(request_id=request_id, event_type=event_type, payload=payload)
        return self.client.publish(self.event_topic, payload=encode(body), qos=1, retain=False)

    def publish_command(self, request_id: str, command: str, payload: Optional[dict[str, Any]] = None) -> mqtt.MQTTMessageInfo:
        body = make_command(request_id=request_id, command=command, payload=payload or {})
        return self.client.publish(self.command_topic, payload=encode(body), qos=1, retain=False)

    def _on_connect(self, client: mqtt.Client, userdata: Any, flags: Any, reason_code: Any, properties: Any = None) -> None:
        if getattr(reason_code, "is_failure", False):
            return
        client.subscribe(self.command_topic, qos=1)
        client.publish(self.availability_topic, payload="online", qos=1, retain=True)
        self._connected.set()

    def _on_disconnect(self, client: mqtt.Client, userdata: Any, disconnect_flags: Any, reason_code: Any, properties: Any = None) -> None:
        self._connected.clear()

    def _on_message(self, client: mqtt.Client, userdata: Any, message: mqtt.MQTTMessage) -> None:
        if message.topic != self.command_topic:
            return
        try:
            body = decode(message.payload)
            request_id = body.get("request_id")
            if not isinstance(request_id, str) or not request_id:
                return
            if not self.idempotency_store.claim(self.device_id, request_id):
                return
            if self.command_handler:
                self.command_handler(body)
        except (TypeError, ValueError, json.JSONDecodeError):
            return
