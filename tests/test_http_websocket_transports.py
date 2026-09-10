from dataclasses import dataclass

import pytest

from src.events.action_envelope import ActionEnvelope
from src.events.http_adapter import HttpActionTransport
from src.events.websocket_adapter import WebSocketActionTransport


@dataclass
class FakePoster:
    calls: list

    def post_json(self, url, payload, headers):
        self.calls.append((url, payload, headers))
        return "http-receipt"


@dataclass
class FakeSocket:
    messages: list

    def send_text(self, payload):
        self.messages.append(payload)
        return "ws-receipt"


def envelope():
    return ActionEnvelope.authorized(
        camera_id="cam-1",
        frame_index=7,
        track_id="track-9",
        person_id="person-42",
        timestamp="2026-09-11T00:00:00+07:00",
    )


def test_http_uses_same_event_id_as_idempotency_key():
    poster = FakePoster([])
    result = HttpActionTransport(poster, "https://device.example/action").send(envelope())
    assert result == "http-receipt"
    assert poster.calls[0][2]["Idempotency-Key"] == envelope().event_id


def test_websocket_preserves_event_identity():
    socket = FakeSocket([])
    result = WebSocketActionTransport(socket).send(envelope())
    assert result == "ws-receipt"
    assert envelope().event_id in socket.messages[0]


@pytest.mark.parametrize("state", ["UNKNOWN", "CANDIDATE", "SPOOF", "ERROR"])
def test_non_authorized_envelopes_are_rejected(state):
    value = envelope()
    invalid = ActionEnvelope(
        schema_version=value.schema_version,
        event_id=value.event_id,
        camera_id=value.camera_id,
        frame_index=value.frame_index,
        track_id=value.track_id,
        person_id=value.person_id,
        authorization_state=state,
        timestamp=value.timestamp,
    )
    with pytest.raises(ValueError):
        HttpActionTransport(FakePoster([]), "https://device.example/action").send(invalid)
    with pytest.raises(ValueError):
        WebSocketActionTransport(FakeSocket([])).send(invalid)
