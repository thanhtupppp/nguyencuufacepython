from uuid import uuid4

from app.event_repository import InMemoryEventRepository


def test_first_insert_then_duplicate_returns_original_payload():
    repo = InMemoryEventRepository()
    event_id = uuid4()
    original = {"event_id": str(event_id), "person_id": "person-1"}
    accepted, stored = repo.put_if_absent(event_id, original)
    assert accepted is True
    assert stored == original

    accepted, stored = repo.put_if_absent(event_id, {"event_id": str(event_id), "person_id": "person-2"})
    assert accepted is False
    assert stored == original
