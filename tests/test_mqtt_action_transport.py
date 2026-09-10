import json

import pytest

from src.events.action_envelope import ActionEnvelope
from src.events.action_dedup import InMemoryActionDeduplicator
from src.events.mqtt_adapter import MqttActionTransport


class FakePublisher:
    def __init__(self):
        self.calls = []

    def publish(self, topic, payload, qos, retain):
        self.calls.append((topic, payload, qos, retain))
        return type("Result", (), {"mid": 17})()


def envelope():
    return ActionEnvelope.authorized(
        camera_id="cam-01",
        frame_index=42,
        track_id="track-7",
        person_id="person-abc",
        timestamp="2026-09-11T00:00:00Z",
    )


def test_mqtt_adapter_preserves_event_and_disables_retain():
    publisher = FakePublisher()
    event = envelope()
    receipt = MqttActionTransport(publisher, "esp32-01", qos=1).send(event)

    assert receipt == 17
    assert len(publisher.calls) == 1
    topic, payload, qos, retain = publisher.calls[0]
    assert topic == "face/esp32-01/actions/v1"
    assert qos == 1
    assert retain is False
    assert json.loads(payload)["event_id"] == event.event_id


def test_non_authorized_event_is_rejected():
    event = envelope()
    rejected = ActionEnvelope(**{**event.to_dict(), "authorization_state": "CANDIDATE"})
    with pytest.raises(ValueError, match="only AUTHORIZED"):
        MqttActionTransport(FakePublisher(), "esp32-01").send(rejected)


def test_duplicate_delivery_executes_physical_action_once():
    guard = InMemoryActionDeduplicator()
    event = envelope()
    calls = []

    first = guard.execute_once(event, "esp32-01", lambda e: calls.append(e.event_id))
    second = guard.execute_once(event, "esp32-01", lambda e: calls.append(e.event_id))
    third = guard.execute_once(event, "esp32-01", lambda e: calls.append(e.event_id))

    assert calls == [event.event_id]
    assert first.executed and not first.duplicate
    assert not second.executed and second.duplicate
    assert not third.executed and third.duplicate
