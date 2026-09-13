"""Explicit application lifecycle state with fail-closed readiness."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from threading import Lock
from typing import Callable

from src.api.readiness import ReadinessGate

logger = logging.getLogger(__name__)

REQUIRED_DEPENDENCIES = ("database", "model_registry", "mqtt", "recognition")


@dataclass
class RuntimeLifecycle:
    """Own dependency readiness and deterministic cleanup callbacks."""

    gate: ReadinessGate = field(default_factory=lambda: ReadinessGate(REQUIRED_DEPENDENCIES))
    _cleanup: list[tuple[str, Callable[[], None]]] = field(default_factory=list)
    _lock: Lock = field(default_factory=Lock)
    started: bool = False
    cleanup_errors: list[str] = field(default_factory=list)

    def mark_ready(self, name: str, detail: str = "initialized") -> None:
        self.gate.set_status(name, True, detail)

    def mark_unready(self, name: str, detail: str = "unavailable") -> None:
        self.gate.set_status(name, False, detail)

    def add_cleanup(self, callback: Callable[[], None], name: str | None = None) -> None:
        resource_name = name or getattr(callback, "__name__", "anonymous_cleanup")
        with self._lock:
            self._cleanup.append((resource_name, callback))

    def startup_complete(self) -> bool:
        self.started = self.gate.ready
        return self.started

    def shutdown(self) -> None:
        with self._lock:
            callbacks = list(reversed(self._cleanup))
            self._cleanup.clear()
            self.cleanup_errors.clear()
        for name, callback in callbacks:
            try:
                callback()
            except Exception as exc:
                detail = type(exc).__name__
                self.cleanup_errors.append(name)
                logger.exception("runtime_cleanup_failed resource=%s error_type=%s", name, detail)
        self.started = False
        for dependency in REQUIRED_DEPENDENCIES:
            self.mark_unready(dependency, "shutdown")
