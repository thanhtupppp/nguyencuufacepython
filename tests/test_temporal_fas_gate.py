from src.fas.temporal_gate import FasObservation, FasState, TemporalFasGate


def obs(frame, track="t1", verified=True, live=True, score=0.9):
    return FasObservation(frame, track, verified, live, score)


def test_three_consecutive_live_confirms():
    gate = TemporalFasGate(live_frames=3)
    assert gate.update(obs(1)).state is FasState.CANDIDATE_LIVE
    assert gate.update(obs(2)).state is FasState.CANDIDATE_LIVE
    assert gate.update(obs(3)).state is FasState.LIVE_CONFIRMED


def test_error_breaks_live_streak():
    gate = TemporalFasGate(live_frames=3)
    gate.update(obs(1))
    gate.update(obs(2))
    assert gate.update(obs(3, verified=False, live=None)).state is FasState.ERROR
    assert gate.update(obs(4)).state is FasState.CANDIDATE_LIVE
    assert gate.update(obs(5)).state is FasState.CANDIDATE_LIVE
    assert gate.update(obs(6)).state is FasState.LIVE_CONFIRMED


def test_spoof_blocks_immediately_by_default():
    gate = TemporalFasGate(live_frames=3, spoof_frames=1)
    assert gate.update(obs(1, live=False)).state is FasState.SPOOF


def test_track_change_does_not_transfer_evidence():
    gate = TemporalFasGate(live_frames=3)
    gate.update(obs(1, track="A"))
    gate.update(obs(2, track="A"))
    assert gate.update(obs(3, track="B")).state is FasState.CANDIDATE_LIVE
    assert gate.update(obs(4, track="B")).state is FasState.CANDIDATE_LIVE
    assert gate.update(obs(5, track="B")).state is FasState.LIVE_CONFIRMED


def test_out_of_order_frame_is_rejected():
    gate = TemporalFasGate()
    gate.update(obs(10))
    try:
        gate.update(obs(9))
    except ValueError as exc:
        assert "monotonically" in str(exc)
    else:
        raise AssertionError("expected out-of-order frame to fail")


def test_confirmed_then_error_does_not_add_live_evidence():
    gate = TemporalFasGate(live_frames=2)
    gate.update(obs(1))
    assert gate.update(obs(2)).state is FasState.LIVE_CONFIRMED
    decision = gate.update(obs(3, verified=False, live=None))
    assert decision.state is FasState.ERROR
    assert decision.live_evidence == 0
