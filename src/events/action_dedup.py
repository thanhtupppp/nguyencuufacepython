"""Consumer-side idempotency guard for physical actions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, MutableSet

from .action_envelope import ActionEnvelope


@dataclass(frozen=True)
class ActionReceipt:
    event_id: str
    device_id: str
    executed: bool
    duplicate: bool


class InMemoryActionDeduplicator:
    """Reference semantics; production should back this with durable storage."""

    def __init__(self) -> None:
        self._seen: MutableSet[str] = set()

    def execute_once(
        self,
        envelope: ActionEnvelope,
        device_id: str,
        action: Callable[[ActionEnvelope], None],
    ) -> ActionReceipt:
        if not envelope.event_id:
            raise ValueError("event_id is required")
        if envelope.event_id in self._seen:
            return ActionReceipt(envelope.event_id, device_id, False, True)
        action(envelope)
        self._seen.add(envelope.event_id)
        return ActionReceipt(envelope.event_id, device_id, True, False)
