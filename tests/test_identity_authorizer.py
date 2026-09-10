from src.fas.identity_authorizer import IdentityConfirmation, authorize_frame
from src.fas.temporal_gate import FasDecision, FasState
from src.tracking.confirmation import TrackIdentityState
from src.fas.authorization import AuthorizationState


def fas(track_id: str, state: FasState, frame: int = 3) -> FasDecision:
    return FasDecision(state, 3 if state is FasState.LIVE_CONFIRMED else 0, 0, track_id, frame)


def identity(track_id: str, state: TrackIdentityState, person_id: str | None) -> IdentityConfirmation:
    return IdentityConfirmation(track_id, state, person_id)


def test_authorizes_only_when_both_gates_confirm_same_track():
    event = authorize_frame(
        frame_index=3,
        camera_id="cam-01",
        fas=fas("t1", FasState.LIVE_CONFIRMED),
        identity=identity("t1", TrackIdentityState.CONFIRMED, "p1"),
    )
    assert event.state is AuthorizationState.AUTHORIZED
    assert event.person_id == "p1"


def test_blocks_live_candidate_identity():
    event = authorize_frame(
        frame_index=2,
        camera_id="cam-01",
        fas=fas("t1", FasState.LIVE_CONFIRMED, 2),
        identity=identity("t1", TrackIdentityState.CANDIDATE, "p1"),
    )
    assert event.state is AuthorizationState.BLOCKED
    assert event.person_id is None


def test_blocks_identity_when_fas_not_confirmed():
    event = authorize_frame(
        frame_index=3,
        camera_id="cam-01",
        fas=fas("t1", FasState.CANDIDATE_LIVE),
        identity=identity("t1", TrackIdentityState.CONFIRMED, "p1"),
    )
    assert event.state is AuthorizationState.BLOCKED
    assert event.person_id is None


def test_blocks_track_mismatch_even_if_both_are_confirmed():
    event = authorize_frame(
        frame_index=3,
        camera_id="cam-01",
        fas=fas("t1", FasState.LIVE_CONFIRMED),
        identity=identity("t2", TrackIdentityState.CONFIRMED, "p2"),
    )
    assert event.state is AuthorizationState.BLOCKED
    assert event.person_id is None
    assert event.reason == "track_id_mismatch"


def test_spoof_never_authorizes_confirmed_identity():
    event = authorize_frame(
        frame_index=3,
        camera_id="cam-01",
        fas=fas("t1", FasState.SPOOF),
        identity=identity("t1", TrackIdentityState.CONFIRMED, "p1"),
    )
    assert event.state is AuthorizationState.BLOCKED
    assert event.person_id is None
