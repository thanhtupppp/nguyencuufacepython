"""Authorization boundary between FAS and face identity actions."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.fas.temporal_gate import FasState


class AuthorizationState(str, Enum):
    BLOCKED = "BLOCKED"
    AUTHORIZED = "AUTHORIZED"


@dataclass(frozen=True)
class AuthorizationDecision:
    state: AuthorizationState
    track_id: str
    person_id: str | None
    reason: str


def authorize_identity(
    *,
    fas_state: FasState,
    track_id: str,
    confirmed_person_id: str | None,
) -> AuthorizationDecision:
    """Allow an identity action only after temporal FAS and identity confirmation.

    FAS never supplies person_id; it only gates whether an already-confirmed
    identity may be acted upon. Missing/conflicting identity evidence fails closed.
    """
    if fas_state is not FasState.LIVE_CONFIRMED:
        return AuthorizationDecision(
            AuthorizationState.BLOCKED, track_id, None, "fas_not_live_confirmed"
        )
    if not confirmed_person_id:
        return AuthorizationDecision(
            AuthorizationState.BLOCKED, track_id, None, "identity_not_confirmed"
        )
    return AuthorizationDecision(
        AuthorizationState.AUTHORIZED,
        track_id,
        confirmed_person_id,
        "fas_and_identity_confirmed",
    )
