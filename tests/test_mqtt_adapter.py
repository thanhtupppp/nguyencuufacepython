from __future__ import annotations

import json

import pytest

from src.devices.mqtt_adapter import MQTTAdapter, MQTTSettings


class FakePublishInfo:
    rc = 0


class FakeClient:
    def __init__(self) -> None:
        self.published: list[tuple[str, object, int, bool]] = []
        self.subscriptions: list[tuple[str, int]] = []
        self.on_connect = None
        self.on_disconnect = None
        self.on_message = None
        self.will = None

    def username_pw_set(self, username, password):
        self.credentials = (username, password)

    def reconnect_delay_set(self, min_delay, max_delay):
        self.reconnect_delays = (min_delay, max_delay)

    def will_set(self, topic, payload, qos, retain):
        self.will = (topic, payload, qos, retain)

    def subscribe(self, topic, qos):
        self.subscriptions.append((topic, qos))
        return FakePublishInfo()

    def publish(self, topic, payload, qos, retain):
        self.published.append((topic, payload, qos, retain))
        return FakePublishInfo()

    def connect(self, host, port, keepalive):
        self.connection = (host, port, keepalive)

    def loop_start(self):
        self.loop_started = True

    def loop_stop(self):
        self.loop_started = False

    def disconnect(self):
        self.disconnected = True


def make_adapter(handler=None):
    client = FakeClient()
    adapter = MQTTAdapter(
        MQTTSettings(host="broker.local", port=1883),
        device_id="cam-01",
        command_handler=handler,
        client=client,
    )
    return adapter, client


def test_adapter_uses_device_scoped_topics_and_will():
    adapter, client = make_adapter()
    assert adapter.command_topic == "face/v1/devices/cam-01/command"
    assert adapter.state_topic == "face/v1/devices/cam-01/state"
    assert adapter.event_topic == "face/v1/devices/cam-01/event"
    assert client.will == (adapter.availability_topic, "offline", 1, True)


def test_connect_subscribes_and_publishes_online():
    adapter, client = make_adapter()
    adapter._on_connect(client, None, {}, 0, None)
    assert client.subscriptions == [(adapter.command_topic, 1)]
    assert client.published[-1] == (adapter.availability_topic, "online", 1, True)
    assert adapter.is_connected


def test_command_redelivery_is_deduplicated():
    received = []
    adapter, client = make_adapter(received.append)
    payload = json.dumps({
        "schema_version": 1,
        "request_id": "req-1",
        "command": "recognize",
        "payload": {"camera_id": "cam-01"},
    }).encode()

    class Message:
        topic = adapter.command_topic
        pass

    message = Message()
    message.payload = payload
    adapter._on_message(client, None, message)
    adapter._on_message(client, None, message)
    assert len(received) == 1


def test_invalid_command_is_ignored():
    received = []
    adapter, client = make_adapter(received.append)

    class Message:
        topic = adapter.command_topic
        payload = b'{"schema_version":999}'

    adapter._on_message(client, None, Message())
    assert received == []


def test_publish_state_is_retained_and_event_is_not():
    adapter, client = make_adapter()
    adapter.publish_state(status="online", firmware_version="1.0.0", model_version="arcface-v1")
    adapter.publish_event(request_id="req-2", event_type="recognition", payload={"person_id": "p-1", "status": "CONFIRMED"})

    assert client.published[-2][0] == adapter.state_topic
    assert client.published[-2][2:] == (1, True)
    assert client.published[-1][0] == adapter.event_topic
    assert client.published[-1][2:] == (1, False)
    assert b"embedding" not in client.published[-1][1]
