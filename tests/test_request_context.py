from src.observability.request_context import get_request_id, new_request_id, set_request_id


def test_request_id_is_bounded_and_preserves_valid_value():
    value = "req-123:abc"
    assert new_request_id(value) == value
    set_request_id(value)
    assert get_request_id() == value


def test_request_id_replaces_unsafe_client_value():
    generated = new_request_id("bad\nheader")
    assert len(generated) <= 128
    assert "\n" not in generated
