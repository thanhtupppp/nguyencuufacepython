"""Explicit application lifecycle state with fail-closed readiness."""
from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Callable

from src.api.readiness import ReadinessGate

REQUIRED_DEPENDENCIES = ("database", "model_registry", "mqtt", "recognition")


@dataclass
class RuntimeLifecycle:
    """Owns dependency readiness and cleanup callbacks without import-time IO."""

    gate: ReadinessGate = field(default_factory=lambda: ReadinessGate(REQUIRED_DEPENDENCIES))
    _cleanup: list[Callable[[], None]] = field(default_factory=list)
    _lock: Lock = field(default_factory=Lock)
    started: bool = False

    def mark_ready(self, name: str, detail: str = "initialized") -> None:
        self.gate.set_status(name, True, detail)

    def mark_unready(self, name: str, detail: str = "unavailable") -> None:
        self.gate.set_status(name, False, detail)

    def add_cleanup(self, callback: Callable[[], None]) -> None:
        with self._lock:
            self._cleanup.append(callback)

    def startup_complete(self) -> bool:
        self.started = self.gate.ready
        return self.started

    def shutdown(self) -> None:
        with self._lock:
            callbacks = list(reversed(self._cleanup))
            self._cleanup.clear()
        for callback in callbacks:
            try:
                callback()
            except Exception:
                # Shutdown is best-effort; never expose cleanup internals to clients.
                continue
        self.started = False
        for name in REQUIRED_DEPENDENCIES:
            self.mark_unready(name, "shutdown")
