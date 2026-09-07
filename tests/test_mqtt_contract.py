import pytest

from src.devices.mqtt_contract import (
    QOS_COMMAND,
    RETAIN_STATE,
    decode,
    encode,
    make_command,
    make_state,
    make_topics,
)


def test_topics_are_device_scoped():
    topics = make_topics("cam-01")
    assert topics.command == "face/v1/devices/cam-01/command"
    assert topics.state == "face/v1/devices/cam-01/state"
    assert topics.event == "face/v1/devices/cam-01/event"
    assert topics.availability == "face/v1/devices/cam-01/availability"


def test_topic_device_id_rejects_wildcards_and_slashes():
    with pytest.raises(ValueError):
        make_topics("cam/01")
    with pytest.raises(ValueError):
        make_topics("cam+#")


def test_command_has_schema_and_request_id_for_idempotency():
    payload = make_command("recognize", "req-123", payload={"camera_id": "cam-01"})
    assert payload == {
        "schema_version": 1,
        "request_id": "req-123",
        "command": "recognize",
        "payload": {"camera_id": "cam-01"},
    }


def test_state_is_safe_to_retain_and_has_device_identity():
    payload = make_state(
        "cam-01",
        status="online",
        firmware_version="1.0.0",
        model_version="arcface-v1",
    )
    assert payload["schema_version"] == 1
    assert payload["device_id"] == "cam-01"
    assert payload["model_version"] == "arcface-v1"
    assert "embedding" not in payload
    assert "image" not in payload
    assert RETAIN_STATE is True
    assert QOS_COMMAND == 1


def test_encode_decode_round_trip():
    original = make_command("health", "req-456")
    assert decode(encode(original)) == original


def test_decode_rejects_unknown_schema():
    with pytest.raises(ValueError):
        decode('{"schema_version": 2}')
