from src.devices.session import DeviceSessionRegistry


def test_session_is_scoped_to_device_and_sequence_is_monotonic() -> None:
    registry = DeviceSessionRegistry()
    session = registry.start("cam-01", session_id="s1")

    touched = registry.touch("s1", "cam-01")
    assert touched.session_id == "s1"
    assert touched.device_id == "cam-01"
    assert touched.sequence == 1
    assert touched.status == "ONLINE"

    closed = registry.close("s1", "cam-01")
    assert closed.status == "OFFLINE"
    assert closed.sequence == 1


def test_same_session_id_cannot_be_used_by_another_device() -> None:
    registry = DeviceSessionRegistry()
    registry.start("cam-01", session_id="shared")

    try:
        registry.touch("shared", "cam-02")
        assert False, "expected device ownership rejection"
    except ValueError as exc:
        assert "does not belong" in str(exc)


def test_invalid_device_topic_characters_are_rejected() -> None:
    registry = DeviceSessionRegistry()
    for device_id in ("cam/01", "cam+01", "cam#01", ""):
        try:
            registry.start(device_id)
            assert False, f"expected rejection for {device_id!r}"
        except ValueError:
            pass
