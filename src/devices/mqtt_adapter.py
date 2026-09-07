"""MQTT v1 transport adapter for edge-device control/state/events.

The adapter deliberately carries compact JSON envelopes only; biometric images and
embeddings stay on HTTP/WebSocket recognition paths and never enter MQTT.
"""

from __future__ import annotations

import json
import ssl
import threading
from dataclasses import dataclass
from typing import Any, Callable, Optional

import paho.mqtt.client as mqtt

from src.devices.mqtt_contract import (
    build_availability_topic,
    build_command_topic,
    build_event_topic,
    build_state_topic,
    encode_command,
    encode_event,
    encode_state,
    validate_device_id,
)


CommandHandler = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class MQTTSettings:
    host: str
    port: int = 8883
    client_id: str = "face-service"
    username: Optional[str] = None
    password: Optional[str] = None
    ca_cert: Optional[str] = None
    client_cert: Optional[str] = None
    client_key: Optional[str] = None
    keepalive: int = 60
    reconnect_min_delay: int = 1
    reconnect_max_delay: int = 30


class MQTTAdapter:
    """Small, testable Paho adapter implementing the repository MQTT v1 contract."""

    def __init__(
        self,
        settings: MQTTSettings,
        device_id: str,
        command_handler: Optional[CommandHandler] = None,
        client: Optional[mqtt.Client] = None,
    ) -> None:
        validate_device_id(device_id)
        self.settings = settings
        self.device_id = device_id
        self.command_handler = command_handler
        self._seen_request_ids: set[str] = set()
        self._lock = threading.Lock()
        self._connected = threading.Event()

        self.client = client or mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=settings.client_id)
        if settings.username is not None:
            self.client.username_pw_set(settings.username, settings.password)
        if settings.ca_cert:
            self.client.tls_set(
                ca_certs=settings.ca_cert,
                certfile=settings.client_cert,
                keyfile=settings.client_key,
                tls_version=ssl.PROTOCOL_TLS_CLIENT,
            )
        self.client.reconnect_delay_set(
            min_delay=settings.reconnect_min_delay,
            max_delay=settings.reconnect_max_delay,
        )
        self.client.will_set(
            build_availability_topic(device_id),
            payload="offline",
            qos=1,
            retain=True,
        )
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message

    @property
    def command_topic(self) -> str:
        return build_command_topic(self.device_id)

    @property
    def state_topic(self) -> str:
        return build_state_topic(self.device_id)

    @property
    def event_topic(self) -> str:
        return build_event_topic(self.device_id)

    @property
    def availability_topic(self) -> str:
        return build_availability_topic(self.device_id)

    def connect(self) -> None:
        self.client.connect(self.settings.host, self.settings.port, self.settings.keepalive)
        self.client.loop_start()

    def disconnect(self) -> None:
        self.client.loop_stop()
        self.client.disconnect()

    def publish_state(self, state: dict[str, Any]) -> mqtt.MQTTMessageInfo:
        payload = encode_state(self.device_id, state)
        return self.client.publish(self.state_topic, payload=payload, qos=1, retain=True)

    def publish_event(self, event: dict[str, Any]) -> mqtt.MQTTMessageInfo:
        payload = encode_event(event)
        return self.client.publish(self.event_topic, payload=payload, qos=1, retain=False)

    def publish_command(self, request_id: str, command: str, payload: Optional[dict[str, Any]] = None) -> mqtt.MQTTMessageInfo:
        body = encode_command(request_id=request_id, command=command, payload=payload or {})
        return self.client.publish(self.command_topic, payload=body, qos=1, retain=False)

    def _on_connect(self, client: mqtt.Client, userdata: Any, flags: Any, reason_code: Any, properties: Any = None) -> None:
        if int(reason_code) != 0:
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
            body = json.loads(message.payload.decode("utf-8"))
            request_id = body.get("request_id")
            if not isinstance(request_id, str) or not request_id:
                return
            with self._lock:
                if request_id in self._seen_request_ids:
                    return
                self._seen_request_ids.add(request_id)
            if self.command_handler:
                self.command_handler(body)
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
            return
