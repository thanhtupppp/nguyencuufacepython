from src.events.action_envelope import ActionEnvelope
from src.events.transport import validate_action_envelope


def test_authorized_envelope_is_deterministic_and_explicit():
    a = ActionEnvelope.authorized(
        camera_id="cam-01",
        frame_index=42,
        track_id="track-7",
        person_id="person-123",
        timestamp="2026-09-10T23:00:00+07:00",
    )
    b = ActionEnvelope.authorized(
        camera_id="cam-01",
        frame_index=42,
        track_id="track-7",
        person_id="person-123",
        timestamp="2026-09-10T23:00:00+07:00",
    )
    assert a.event_id == b.event_id
    assert a.authorization_state == "AUTHORIZED"
    assert a.to_dict()["person_id"] == "person-123"
    validate_action_envelope(a)


def test_changed_coordinates_change_event_id():
    a = ActionEnvelope.authorized(
        camera_id="cam-01", frame_index=42, track_id="track-7",
        person_id="person-123", timestamp="t",
    )
    b = ActionEnvelope.authorized(
        camera_id="cam-01", frame_index=43, track_id="track-7",
        person_id="person-123", timestamp="t",
    )
    assert a.event_id != b.event_id


def test_non_authorized_state_is_rejected():
    a = ActionEnvelope.authorized(
        camera_id="cam-01", frame_index=1, track_id="track-1",
        person_id="person-1", timestamp="t",
    )
    object.__setattr__(a, "authorization_state", "CANDIDATE")
    try:
        validate_action_envelope(a)
    except ValueError as exc:
        assert "AUTHORIZED" in str(exc)
    else:
        raise AssertionError("non-authorized envelope was accepted")
