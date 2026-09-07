"""Regression tests for appearance-aware tracking and weighted temporal fusion."""

import numpy as np

from src.tracking.tracker import DetectionItem, FaceTracker, TemporalVotingEngine, Tracklet


def test_weighted_temporal_fusion_prefers_strong_evidence():
    engine = TemporalVotingEngine(window_size=5, min_consensus_ratio=0.60, min_frames=3)
    track = Tracklet(track_id=10, start_frame=0, last_frame=0, bbox=[0, 0, 50, 50])
    for frame, sim in enumerate((0.92, 0.91)):
        track.add_detection(DetectionItem(frame_idx=frame, bbox=[0, 0, 50, 50], score=0.98,
                                          quality_score=0.98, predicted_id="A", similarity=sim))
    for frame, sim in enumerate((0.60, 0.61, 0.59), start=2):
        track.add_detection(DetectionItem(frame_idx=frame, bbox=[0, 0, 50, 50], score=0.70,
                                          quality_score=0.60, predicted_id="B", similarity=sim))
    result = engine.vote(track)
    assert result.status == "CONFIRMED_MATCH"
    assert result.person_id == "A"
    assert result.vote_breakdown == {"A": 2, "B": 3}
    assert result.weighted_breakdown["A"] > result.weighted_breakdown["B"]


def test_tracker_uses_embedding_when_iou_is_weak():
    tracker = FaceTracker(iou_threshold=0.30, appearance_weight=0.50,
                          appearance_threshold=0.80, min_hits_to_activate=1)
    emb = np.zeros(512, dtype=np.float32); emb[0] = 1.0
    first = DetectionItem(frame_idx=0, bbox=[0, 0, 50, 50], score=0.95, embedding=emb)
    assert tracker.update([first], 0)[0].track_id == 1

    moved = DetectionItem(frame_idx=1, bbox=[45, 0, 95, 50], score=0.95, embedding=emb.copy())
    active = tracker.update([moved], 1)
    assert active[0].track_id == 1
    assert active[0].hits == 2


def test_lost_track_can_be_recovered_by_embedding():
    tracker = FaceTracker(iou_threshold=0.90, appearance_threshold=0.80,
                          recovery_embedding_threshold=0.90, min_hits_to_activate=1)
    emb = np.zeros(512, dtype=np.float32); emb[7] = 1.0
    first = DetectionItem(frame_idx=0, bbox=[0, 0, 50, 50], score=0.95, embedding=emb)
    assert tracker.update([first], 0)[0].track_id == 1

    # Force a miss: the original track enters the bounded lost pool.
    assert tracker.update([], 1) == []
    assert 1 in tracker.lost_tracklets

    recovered = DetectionItem(frame_idx=2, bbox=[300, 200, 350, 250], score=0.95, embedding=emb.copy())
    active = tracker.update([recovered], 2)
    assert active[0].track_id == 1
    assert active[0].hits == 2
    assert 1 not in tracker.lost_tracklets


def test_low_quality_false_identity_does_not_overpower_strong_frame():
    engine = TemporalVotingEngine(window_size=3, min_consensus_ratio=0.60, min_frames=2)
    track = Tracklet(track_id=11, start_frame=0, last_frame=0, bbox=[0, 0, 50, 50])
    track.add_detection(DetectionItem(frame_idx=0, bbox=[0, 0, 50, 50], score=0.99,
                                     quality_score=0.99, predicted_id="A", similarity=0.90))
    track.add_detection(DetectionItem(frame_idx=1, bbox=[0, 0, 50, 50], score=0.30,
                                     quality_score=0.10, predicted_id="B", similarity=0.99))
    result = engine.vote(track)
    assert result.status == "AMBIGUOUS"
    assert result.person_id is None
