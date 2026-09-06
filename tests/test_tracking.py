"""
Unit tests for FaceTracker and TemporalVotingEngine.
"""

import numpy as np
import pytest
from src.tracking.tracker import (
    compute_iou,
    compute_iou_matrix,
    DetectionItem,
    Tracklet,
    FaceTracker,
    TemporalVotingEngine,
    VotingResult,
)


def test_compute_iou():
    # Identical boxes
    box_a = [10.0, 10.0, 50.0, 50.0]
    assert np.isclose(compute_iou(box_a, box_a), 1.0)

    # Disjoint boxes
    box_b = [100.0, 100.0, 150.0, 150.0]
    assert np.isclose(compute_iou(box_a, box_b), 0.0)

    # Partial overlap
    box_c = [30.0, 30.0, 70.0, 70.0]
    iou = compute_iou(box_a, box_c)
    assert 0.1 < iou < 0.5


def test_tracklet_best_frame_selection():
    """Verify tracklet automatically caches the observation with highest quality score."""
    tr = Tracklet(track_id=1, start_frame=0, last_frame=0, bbox=[10, 10, 50, 50])

    item_low = DetectionItem(frame_idx=0, bbox=[10, 10, 50, 50], score=0.9, quality_score=0.45)
    item_high = DetectionItem(frame_idx=1, bbox=[12, 11, 52, 51], score=0.95, quality_score=0.92)
    item_med = DetectionItem(frame_idx=2, bbox=[14, 12, 54, 52], score=0.91, quality_score=0.68)

    tr.add_detection(item_low)
    assert tr.best_quality_score == 0.45
    assert tr.best_item == item_low

    tr.add_detection(item_high)
    assert tr.best_quality_score == 0.92
    assert tr.best_item == item_high

    tr.add_detection(item_med)
    # Best quality should still be item_high (0.92)
    assert tr.best_quality_score == 0.92
    assert tr.best_item == item_high


def test_face_tracker_association():
    """Verify face tracker maintains track_id across consecutive frames."""
    tracker = FaceTracker(iou_threshold=0.3, min_hits_to_activate=2)

    # Frame 0: One face detected
    det0 = [DetectionItem(frame_idx=0, bbox=[100.0, 100.0, 180.0, 180.0], score=0.95)]
    active0 = tracker.update(det0, frame_idx=0)
    # Hit count = 1, min_hits_to_activate = 2 -> not returned yet
    assert len(active0) == 0

    # Frame 1: Slightly moved face (high IoU)
    det1 = [DetectionItem(frame_idx=1, bbox=[104.0, 102.0, 184.0, 182.0], score=0.96)]
    active1 = tracker.update(det1, frame_idx=1)
    assert len(active1) == 1
    assert active1[0].track_id == 1
    assert active1[0].hits == 2

    # Frame 2: Same face + a brand new person enters scene
    det2 = [
        DetectionItem(frame_idx=2, bbox=[108.0, 104.0, 188.0, 184.0], score=0.94),
        DetectionItem(frame_idx=2, bbox=[400.0, 200.0, 480.0, 280.0], score=0.92),
    ]
    active2 = tracker.update(det2, frame_idx=2)
    # Face 1 remains active; new face 2 has hit=1
    assert len(active2) == 1
    assert active2[0].track_id == 1


def test_temporal_voting_suppresses_noise():
    """
    Simulate real-world scenario:
    Out of 5 frames, 4 frames correctly predict 'person_001' (similarity ~0.75).
    1 frame encounters blur and falsely predicts 'person_002' (similarity 0.61).
    Consensus voting must suppress the single-frame noise and confirm 'person_001'.
    """
    engine = TemporalVotingEngine(window_size=5, min_consensus_ratio=0.60, min_frames=3)
    tr = Tracklet(track_id=1, start_frame=0, last_frame=0, bbox=[10, 10, 50, 50])

    # 4 frames person_001
    for i in range(4):
        tr.add_detection(DetectionItem(
            frame_idx=i, bbox=[10, 10, 50, 50], score=0.95,
            predicted_id="person_001", similarity=0.75 + i * 0.01, is_valid_quality=True
        ))

    # 1 corrupted frame person_002
    tr.add_detection(DetectionItem(
        frame_idx=4, bbox=[10, 10, 50, 50], score=0.90,
        predicted_id="person_002", similarity=0.61, is_valid_quality=True
    ))

    res = engine.vote(tr)
    assert res.status == "CONFIRMED_MATCH"
    assert res.person_id == "person_001"
    assert res.consensus_ratio == 0.80  # 4/5 = 80%
    assert res.vote_breakdown == {"person_001": 4, "person_002": 1}


def test_temporal_voting_ambiguous_on_split_vote():
    """If votes are split (e.g., 2 votes A, 2 votes B, 1 vote C), decision must be AMBIGUOUS."""
    engine = TemporalVotingEngine(window_size=5, min_consensus_ratio=0.60, min_frames=3)
    tr = Tracklet(track_id=2, start_frame=0, last_frame=0, bbox=[10, 10, 50, 50])

    tr.add_detection(DetectionItem(frame_idx=0, bbox=[10, 10, 50, 50], score=0.9, predicted_id="person_A", similarity=0.65))
    tr.add_detection(DetectionItem(frame_idx=1, bbox=[10, 10, 50, 50], score=0.9, predicted_id="person_A", similarity=0.66))
    tr.add_detection(DetectionItem(frame_idx=2, bbox=[10, 10, 50, 50], score=0.9, predicted_id="person_B", similarity=0.64))
    tr.add_detection(DetectionItem(frame_idx=3, bbox=[10, 10, 50, 50], score=0.9, predicted_id="person_B", similarity=0.65))
    tr.add_detection(DetectionItem(frame_idx=4, bbox=[10, 10, 50, 50], score=0.9, predicted_id="person_C", similarity=0.62))

    res = engine.vote(tr)
    # Highest candidate has 2/5 = 40% < 60% min_consensus -> AMBIGUOUS
    assert res.status == "AMBIGUOUS"
    assert res.person_id is None
    assert res.consensus_ratio == 0.40


def test_temporal_voting_pending_when_insufficient_frames():
    """If fewer than min_frames observations exist, status must remain PENDING."""
    engine = TemporalVotingEngine(window_size=5, min_consensus_ratio=0.60, min_frames=3)
    tr = Tracklet(track_id=3, start_frame=0, last_frame=0, bbox=[10, 10, 50, 50])

    # Only 2 frames
    tr.add_detection(DetectionItem(frame_idx=0, bbox=[10, 10, 50, 50], score=0.9, predicted_id="person_A", similarity=0.7))
    tr.add_detection(DetectionItem(frame_idx=1, bbox=[10, 10, 50, 50], score=0.9, predicted_id="person_A", similarity=0.7))

    res = engine.vote(tr)
    assert res.status == "PENDING"
    assert res.frame_count == 2
