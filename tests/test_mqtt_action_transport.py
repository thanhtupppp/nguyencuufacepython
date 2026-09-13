import asyncio
import json

import pytest

from src.events.action_envelope import ActionEnvelope
from src.events.action_dedup import InMemoryActionDeduplicator
from src.events.mqtt_adapter import AsyncMqttActionTransport, MqttActionTransport


class FakePublisher:
    def __init__(self, failures=0, error=TimeoutError):
        self.calls = []
        self.failures = failures
        self.error = error

    def publish(self, topic, payload, qos, retain):
        self.calls.append((topic, payload, qos, retain))
        if len(self.calls) <= self.failures:
            raise self.error("transient")
        return type("Result", (), {"mid": 17})()


class AsyncFakePublisher:
    def __init__(self, failures=0):
        self.calls = []
        self.failures = failures

    async def publish(self, topic, payload, qos, retain):
        self.calls.append((topic, payload, qos, retain))
        if len(self.calls) <= self.failures:
            raise TimeoutError("transient")
        return type("Result", (), {"mid": 19})()


def envelope():
    return ActionEnvelope.authorized(
        camera_id="cam-01", frame_index=42, track_id="track-7", person_id="person-abc", timestamp="2026-09-11T00:00:00Z"
    )


def test_mqtt_adapter_preserves_event_and_disables_retain():
    publisher = FakePublisher()
    event = envelope()
    receipt = MqttActionTransport(publisher, "esp32-01", qos=1).send(event)
    assert receipt == 17
    assert json.loads(publisher.calls[0][1])["event_id"] == event.event_id
    assert publisher.calls[0][3] is False


def test_non_authorized_event_is_rejected_without_publish():
    event = envelope()
    rejected = ActionEnvelope(**{**event.to_dict(), "authorization_state": "CANDIDATE"})
    publisher = FakePublisher()
    with pytest.raises(ValueError, match="only AUTHORIZED"):
        MqttActionTransport(publisher, "esp32-01").send(rejected)
    assert publisher.calls == []


@pytest.mark.parametrize("max_retries, failures, expected_calls", [(0, 3, 1), (1, 3, 2), (2, 3, 3), (2, 1, 2)])
def test_retry_count_is_bounded(max_retries, failures, expected_calls):
    publisher = FakePublisher(failures=failures)
    transport = MqttActionTransport(publisher, "esp32-01", max_retries=max_retries, retry_delay_s=0)
    with pytest.raises(RuntimeError):
        transport.send(envelope()) if failures > max_retries else transport.send(envelope())
    assert len(publisher.calls) == expected_calls


def test_retry_succeeds_after_transient_failure():
    publisher = FakePublisher(failures=2)
    transport = MqttActionTransport(publisher, "esp32-01", max_retries=2, retry_delay_s=0)
    assert transport.send(envelope()) == 17
    assert len(publisher.calls) == 3


def test_non_transient_payload_error_is_not_retried(monkeypatch):
    publisher = FakePublisher()
    transport = MqttActionTransport(publisher, "esp32-01", max_retries=2, retry_delay_s=0)
    monkeypatch.setattr("src.events.mqtt_adapter.validate_action_envelope", lambda event: (_ for _ in ()).throw(ValueError("invalid")))
    with pytest.raises(ValueError):
        transport.send(envelope())
    assert publisher.calls == []


def test_async_transport_uses_async_backoff():
    publisher = AsyncFakePublisher(failures=1)
    transport = AsyncMqttActionTransport(publisher, "esp32-01", max_retries=1, retry_delay_s=0)
    assert asyncio.run(transport.send(envelope())) == 19
    assert len(publisher.calls) == 2


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
