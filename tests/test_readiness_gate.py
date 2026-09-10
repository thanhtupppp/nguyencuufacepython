from src.api.readiness import ReadinessGate


def test_readiness_is_fail_closed_until_every_dependency_is_ready():
    gate = ReadinessGate(("scrfd", "arcface", "alignment", "fas", "vector_db", "model_provenance"))
    assert gate.ready is False
    gate.set_status("scrfd", True)
    gate.set_status("arcface", True)
    gate.set_status("alignment", True)
    gate.set_status("fas", True)
    gate.set_status("vector_db", True)
    assert gate.ready is False
    gate.set_status("model_provenance", True)
    assert gate.ready is True


def test_unknown_dependency_cannot_be_silently_added():
    gate = ReadinessGate(("scrfd",))
    try:
        gate.set_status("arcface", True)
    except KeyError:
        return
    raise AssertionError("unknown readiness dependency was accepted")
