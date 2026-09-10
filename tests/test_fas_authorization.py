from src.fas.authorization import AuthorizationState, authorize_identity
from src.fas.temporal_gate import FasState


def test_live_and_confirmed_identity_authorize():
    decision = authorize_identity(
        fas_state=FasState.LIVE_CONFIRMED,
        track_id="track-1",
        confirmed_person_id="person-42",
    )
    assert decision.state is AuthorizationState.AUTHORIZED
    assert decision.person_id == "person-42"


def test_fas_not_live_blocks_identity():
    decision = authorize_identity(
        fas_state=FasState.SPOOF,
        track_id="track-1",
        confirmed_person_id="person-42",
    )
    assert decision.state is AuthorizationState.BLOCKED
    assert decision.person_id is None


def test_unconfirmed_identity_blocks_even_when_live():
    decision = authorize_identity(
        fas_state=FasState.LIVE_CONFIRMED,
        track_id="track-1",
        confirmed_person_id=None,
    )
    assert decision.state is AuthorizationState.BLOCKED
    assert decision.person_id is None
