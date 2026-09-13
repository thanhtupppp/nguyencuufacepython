from src.contracts.error import error_envelope
from src.observability.request_context import set_request_id


def test_error_envelope_is_sanitized_and_correlated():
    set_request_id("req-err-1")
    value = error_envelope("DEPENDENCY_UNAVAILABLE", "Service dependency unavailable")
    assert set(value) == {"code", "message", "request_id", "timestamp"}
    assert value["request_id"] == "req-err-1"
    assert "password" not in value["message"].lower()
