"""Real-broker MQTT integration tests.

Run explicitly with MQTT_INTEGRATION=1 after starting the local Mosquitto
service from docker-compose.yml. These tests intentionally use plaintext on
localhost only; production broker security is covered by adapter configuration
and must use TLS/authentication/ACLs.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid

import pytest
import paho.mqtt.client as mqtt

from src.devices.mqtt_adapter import MQTTAdapter, MQTTSettings


pytestmark = pytest.mark.skipif(
    os.getenv("MQTT_INTEGRATION") != "1",
    reason="set MQTT_INTEGRATION=1 to run against a real MQTT broker",
)


@pytest.mark.integration
def test_command_round_trip_and_request_id_deduplication() -> None:
    host = os.getenv("MQTT_HOST", "127.0.0.1")
    port = int(os.getenv("MQTT_PORT", "1883"))
    device_id = f"it-{uuid.uuid4().hex[:8]}"
    request_id = str(uuid.uuid4())
    received: list[dict] = []
    done = threading.Event()

    def handler(body: dict) -> None:
        received.append(body)
        done.set()

    adapter = MQTTAdapter(
        MQTTSettings(host=host, port=port, client_id=f"service-{uuid.uuid4().hex[:8]}"),
        device_id=device_id,
        command_handler=handler,
    )

    publisher = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"pub-{uuid.uuid4().hex[:8]}")
    try:
        adapter.connect()
        assert adapter._connected.wait(5), "adapter did not connect"

        publisher.connect(host, port, 60)
        publisher.loop_start()
        time.sleep(0.2)

        payload = json.dumps({
            "schema_version": "1.0",
            "request_id": request_id,
            "command": "ping",
            "payload": {},
        })
        publisher.publish(adapter.command_topic, payload=payload, qos=1, retain=False).wait_for_publish(5)
        assert done.wait(5), "command was not delivered"

        # Same request_id simulates QoS-1 duplicate delivery.
        publisher.publish(adapter.command_topic, payload=payload, qos=1, retain=False).wait_for_publish(5)
        time.sleep(0.5)
        assert len(received) == 1
        assert received[0]["request_id"] == request_id
    finally:
        publisher.loop_stop()
        publisher.disconnect()
        adapter.disconnect()


@pytest.mark.integration
def test_state_is_retained_and_availability_is_retained() -> None:
    host = os.getenv("MQTT_HOST", "127.0.0.1")
    port = int(os.getenv("MQTT_PORT", "1883"))
    device_id = f"it-{uuid.uuid4().hex[:8]}"
    adapter = MQTTAdapter(
        MQTTSettings(host=host, port=port, client_id=f"service-{uuid.uuid4().hex[:8]}"),
        device_id=device_id,
    )
    subscriber = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"sub-{uuid.uuid4().hex[:8]}")
    messages: list[tuple[str, str, bool]] = []
    done = threading.Event()

    def on_message(client, userdata, message):
        messages.append((message.topic, message.payload.decode(), bool(message.retain)))
        if len(messages) >= 2:
            done.set()

    subscriber.on_message = on_message
    try:
        adapter.connect()
        assert adapter._connected.wait(5)
        adapter.publish_state(status="online", firmware_version="test", camera_id="cam-01").wait_for_publish(5)

        subscriber.connect(host, port, 60)
        subscriber.subscribe(adapter.state_topic, qos=1)
        subscriber.subscribe(adapter.availability_topic, qos=1)
        subscriber.on_message = on_message
        subscriber.loop_start()
        assert done.wait(5), "retained state/availability not observed"
        assert any(topic == adapter.state_topic and retained for topic, _, retained in messages)
        assert any(topic == adapter.availability_topic and retained for topic, _, retained in messages)
    finally:
        subscriber.loop_stop()
        subscriber.disconnect()
        adapter.disconnect()
