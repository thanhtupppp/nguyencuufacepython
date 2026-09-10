"""Single-event authorization boundary for face recognition actions.

The authorizer fuses two independent, already-derived facts for one
camera-local track: temporal FAS liveness and temporal identity confirmation.
It never computes a face similarity score and never assigns person_id.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.fas.authorization import AuthorizationDecision, AuthorizationState, authorize_identity
from src.fas.temporal_gate import FasDecision, FasState
from src.tracking.confirmation import TrackIdentityState


@dataclass(frozen=True)
class IdentityConfirmation:
    track_id: str
    state: TrackIdentityState
    person_id: str | None


@dataclass(frozen=True)
class AuthorizationEvent:
    frame_index: int
    camera_id: str
    track_id: str
    person_id: str | None
    state: AuthorizationState
    reason: str


def authorize_frame(
    *,
    frame_index: int,
    camera_id: str,
    fas: FasDecision,
    identity: IdentityConfirmation,
) -> AuthorizationEvent:
    """Create one fail-closed authorization event for a single frame.

    A confirmed identity from a different track can never be authorized.
    ``person_id`` is emitted only when both temporal gates agree on the same
    track. The function is deterministic and contains no model inference.
    """
    if fas.track_id != identity.track_id:
        decision = AuthorizationDecision(
            AuthorizationState.BLOCKED,
            fas.track_id,
            None,
            "track_id_mismatch",
        )
    else:
        confirmed_person_id = (
            identity.person_id
            if identity.state is TrackIdentityState.CONFIRMED
            else None
        )
        decision = authorize_identity(
            fas_state=fas.state,
            track_id=fas.track_id,
            confirmed_person_id=confirmed_person_id,
        )

    return AuthorizationEvent(
        frame_index=frame_index,
        camera_id=camera_id,
        track_id=decision.track_id,
        person_id=decision.person_id,
        state=decision.state,
        reason=decision.reason,
    )
