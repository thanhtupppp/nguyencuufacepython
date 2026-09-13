from src.events.action_dedup import InMemoryActionDeduplicator
from src.events.action_envelope import ActionEnvelope


def envelope():
    return ActionEnvelope.authorized(
        camera_id="cam-01",
        frame_index=42,
        track_id="track-7",
        person_id="person-abc",
        timestamp="2026-09-13T00:00:00Z",
    )


def test_duplicate_action_is_terminal_and_executes_once():
    guard = InMemoryActionDeduplicator()
    calls = []
    event = envelope()
    first = guard.execute_once(event, "esp32-01", lambda e: calls.append(e.event_id))
    second = guard.execute_once(event, "esp32-01", lambda e: calls.append(e.event_id))
    assert first.executed is True
    assert second.duplicate is True
    assert calls == [event.event_id]
