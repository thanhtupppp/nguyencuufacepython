"""MQTT v1 transport adapter for edge-device control/state/events."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Callable, Optional

import paho.mqtt.client as mqtt

from src.devices.idempotency import IdempotencyStore, TTLMemoryIdempotencyStore
from src.devices.mqtt_contract import decode, encode, make_command, make_event, make_state, make_topics

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
    """Paho adapter implementing MQTT v1 with injectable idempotency storage.

    The default store is bounded/TTL and intended only for one worker. A shared
    PostgresIdempotencyStore should be injected for horizontally scaled workers.
    """

    def __init__(
        self,
        settings: MQTTSettings,
        device_id: str,
        command_handler: Optional[CommandHandler] = None,
        client: Optional[mqtt.Client] = None,
        idempotency_store: Optional[IdempotencyStore] = None,
    ) -> None:
        self.settings = settings
        self.device_id = device_id
        self.topics = make_topics(device_id)
        self.command_handler = command_handler
        self.idempotency_store = idempotency_store or TTLMemoryIdempotencyStore()
        self._connected = threading.Event()

        self.client = client or mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=settings.client_id)
        if settings.username is not None:
            self.client.username_pw_set(settings.username, settings.password)
        if settings.ca_cert:
            self.client.tls_set(ca_certs=settings.ca_cert, certfile=settings.client_cert, keyfile=settings.client_key)
        self.client.reconnect_delay_set(min_delay=settings.reconnect_min_delay, max_delay=settings.reconnect_max_delay)
        self.client.will_set(self.topics.availability, payload="offline", qos=1, retain=True)
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message

    @property
    def command_topic(self) -> str:
        return self.topics.command

    @property
    def state_topic(self) -> str:
        return self.topics.state

    @property
    def event_topic(self) -> str:
        return self.topics.event

    @property
    def availability_topic(self) -> str:
        return self.topics.availability

    @property
    def is_connected(self) -> bool:
        return self._connected.is_set()

    def connect(self) -> None:
        self.client.connect(self.settings.host, self.settings.port, self.settings.keepalive)
        self.client.loop_start()

    def disconnect(self) -> None:
        self.client.loop_stop()
        self.client.disconnect()

    def publish_state(self, *, status: str, firmware_version: str, model_version: str | None = None, camera_id: str | None = None) -> mqtt.MQTTMessageInfo:
        body = make_state(self.device_id, status=status, firmware_version=firmware_version, model_version=model_version, camera_id=camera_id)
        return self.client.publish(self.state_topic, payload=encode(body), qos=1, retain=True)

    def publish_event(self, *, request_id: str, event_type: str, payload: Optional[dict[str, Any]] = None) -> mqtt.MQTTMessageInfo:
        body = make_event(request_id=request_id, event_type=event_type, payload=payload)
        return self.client.publish(self.event_topic, payload=encode(body), qos=1, retain=False)

    def publish_command(self, request_id: str, command: str, payload: Optional[dict[str, Any]] = None) -> mqtt.MQTTMessageInfo:
        body = make_command(command=command, request_id=request_id, payload=payload or {})
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
        except ValueError:
            return
