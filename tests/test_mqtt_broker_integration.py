import json
import os
import threading
from uuid import uuid4

import paho.mqtt.client as mqtt
import pytest

from src.devices.mqtt_contract import decode

pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    os.getenv("MQTT_INTEGRATION") != "1",
    reason="set MQTT_INTEGRATION=1 with a reachable test broker",
)
def test_canonical_event_round_trip_contract() -> None:
    """Exercise a real broker; never silently replace it with a fake."""
    host = os.environ["MQTT_HOST"]
    port = int(os.getenv("MQTT_PORT", "8883"))
    topic = os.getenv("MQTT_EVENT_TOPIC", "face/v1/devices/integration/event")
    timeout = float(os.getenv("MQTT_TEST_TIMEOUT", "10"))
    username = os.getenv("MQTT_USERNAME")
    password = os.getenv("MQTT_PASSWORD")
    ca_cert = os.getenv("MQTT_CA_CERT")

    received: list[dict] = []
    done = threading.Event()

    subscriber = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"it-sub-{uuid4()}")
    publisher = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"it-pub-{uuid4()}")
    for client in (subscriber, publisher):
        if username:
            client.username_pw_set(username, password)
        if ca_cert:
            client.tls_set(ca_certs=ca_cert)

    def on_message(_client, _userdata, message):
        body = decode(message.payload)
        received.append(body)
        done.set()

    subscriber.on_message = on_message
    subscriber.connect(host, port, 60)
    subscriber.subscribe(topic, qos=1)
    subscriber.loop_start()
    publisher.connect(host, port, 60)
    publisher.loop_start()

    payload = {
        "schema_version": 1,
        "event_id": str(uuid4()),
        "state": "RECOGNITION_CONFIRMED",
        "camera_id": "cam-integration",
        "device_id": "device-integration",
        "track_id": "track-1",
        "person_id": "person-1",
        "model_version": "integration-model",
        "embedding_version": "integration-embedding",
        "quality": 0.95,
        "liveness": 0.99,
        "similarity": 0.88,
        "margin": 0.12,
        "frames_confirmed": 3,
        "timestamp": "2026-09-09T20:00:00+07:00",
    }

    try:
        info = publisher.publish(topic, json.dumps(payload, separators=(",", ":")), qos=1, retain=False)
        assert info.wait_for_publish(timeout=timeout)
        assert info.is_published()
        assert done.wait(timeout)
        assert received == [payload]
        assert "embedding" not in received[0]
        assert "image" not in received[0]
    finally:
        publisher.loop_stop()
        subscriber.loop_stop()
        publisher.disconnect()
        subscriber.disconnect()
