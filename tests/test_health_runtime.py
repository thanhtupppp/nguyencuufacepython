from src.health.runtime import REQUIRED_DEPENDENCIES, RuntimeLifecycle


def test_runtime_defaults_to_fail_closed():
    runtime = RuntimeLifecycle()
    assert runtime.started is False
    assert runtime.gate.ready is False
    assert tuple(runtime.gate.snapshot()) == REQUIRED_DEPENDENCIES


def test_runtime_startup_requires_every_dependency():
    runtime = RuntimeLifecycle()
    runtime.mark_ready("database")
    runtime.mark_ready("model_registry")
    runtime.mark_ready("mqtt")
    assert runtime.startup_complete() is False
    runtime.mark_ready("recognition")
    assert runtime.startup_complete() is True


def test_shutdown_closes_registered_resources_and_resets_readiness():
    runtime = RuntimeLifecycle()
    calls = []
    runtime.add_cleanup(lambda: calls.append("closed"))
    for dependency in REQUIRED_DEPENDENCIES:
        runtime.mark_ready(dependency)
    runtime.startup_complete()
    runtime.shutdown()
    assert calls == ["closed"]
    assert runtime.started is False
    assert runtime.gate.ready is False
