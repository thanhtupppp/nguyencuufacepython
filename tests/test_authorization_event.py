import pytest

from src.events.authorization_event import authorization_event_id


def test_event_id_is_deterministic():
    a = authorization_event_id(camera_id="cam-01", frame_index=42, track_id="t7", person_id="p9")
    b = authorization_event_id(camera_id="cam-01", frame_index=42, track_id="t7", person_id="p9")
    assert a == b
    assert len(a) == 64


def test_event_id_changes_when_identity_coordinates_change():
    base = authorization_event_id(camera_id="cam-01", frame_index=42, track_id="t7", person_id="p9")
    assert authorization_event_id(camera_id="cam-01", frame_index=43, track_id="t7", person_id="p9") != base
    assert authorization_event_id(camera_id="cam-01", frame_index=42, track_id="t8", person_id="p9") != base
    assert authorization_event_id(camera_id="cam-01", frame_index=42, track_id="t7", person_id="p8") != base


def test_event_id_rejects_missing_identity_coordinates():
    with pytest.raises(ValueError):
        authorization_event_id(camera_id="", frame_index=1, track_id="t1", person_id="p1")
    with pytest.raises(ValueError):
        authorization_event_id(camera_id="cam", frame_index=1, track_id="", person_id="p1")
    with pytest.raises(ValueError):
        authorization_event_id(camera_id="cam", frame_index=1, track_id="t1", person_id="")
