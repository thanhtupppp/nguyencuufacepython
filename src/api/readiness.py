"""Dependency-aware readiness for the face-recognition service."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class DependencyStatus:
    name: str
    ready: bool
    detail: str = ""


class ReadinessGate:
    """Fail-closed gate: every required dependency must explicitly be ready."""

    def __init__(self, required: tuple[str, ...]) -> None:
        self._required = required
        self._status: dict[str, DependencyStatus] = {
            name: DependencyStatus(name=name, ready=False, detail="not_initialized")
            for name in required
        }

    def set_status(self, name: str, ready: bool, detail: str = "") -> None:
        if name not in self._status:
            raise KeyError(f"unknown readiness dependency: {name}")
        self._status[name] = DependencyStatus(name=name, ready=ready, detail=detail)

    @property
    def ready(self) -> bool:
        return all(item.ready for item in self._status.values())

    def snapshot(self) -> Mapping[str, DependencyStatus]:
        return dict(self._status)

    def as_dict(self) -> dict[str, object]:
        return {
            "ready": self.ready,
            "dependencies": {
                name: {"ready": item.ready, "detail": item.detail}
                for name, item in self._status.items()
            },
        }
