from src.tracking.confirmation import (
    ConfirmationConfig,
    IdentityObservation,
    TrackConfirmation,
    TrackIdentityState,
)


def obs(pid, accepted=True, score=0.9):
    return IdentityObservation(person_id=pid, accepted=accepted, score=score)


def test_confirmation_requires_consecutive_same_person():
    track = TrackConfirmation("cam1-track1", ConfirmationConfig(min_consecutive=3))
    assert track.observe(1, obs("p1")) == TrackIdentityState.CANDIDATE
    assert track.observe(2, obs("p1")) == TrackIdentityState.CANDIDATE
    assert track.observe(3, obs("p1")) == TrackIdentityState.CONFIRMED
    assert track.person_id == "p1"


def test_conflicting_candidate_does_not_confirm_wrong_person():
    track = TrackConfirmation("cam1-track2", ConfirmationConfig(min_consecutive=3))
    assert track.observe(1, obs("p1")) == TrackIdentityState.CANDIDATE
    assert track.observe(2, obs("p2")) == TrackIdentityState.CANDIDATE
    assert track.person_id is None
    assert track.candidate_person_id == "p2"


def test_unknown_observation_resets_unconfirmed_candidate():
    track = TrackConfirmation("cam1-track3", ConfirmationConfig(min_consecutive=3))
    track.observe(1, obs("p1"))
    assert track.observe(2, obs(None, accepted=False)) == TrackIdentityState.UNKNOWN
    assert track.candidate_person_id is None
    assert track.consecutive_hits == 0


def test_confirmed_identity_is_not_silently_replaced():
    track = TrackConfirmation("cam1-track4", ConfirmationConfig(min_consecutive=2))
    track.observe(1, obs("p1"))
    assert track.observe(2, obs("p1")) == TrackIdentityState.CONFIRMED
    assert track.observe(3, obs("p2")) == TrackIdentityState.CONFIRMED
    assert track.person_id == "p1"


def test_large_gap_expires_track():
    track = TrackConfirmation("cam1-track5", ConfirmationConfig(min_consecutive=2, max_gap_frames=2))
    track.observe(1, obs("p1"))
    assert track.expire(5) == TrackIdentityState.LOST
    assert track.person_id is None
