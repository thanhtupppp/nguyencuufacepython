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


def test_compute_iou_matrix_vectorized():
    """Verify vectorized compute_iou_matrix behaves identically to pairwise compute_iou."""
    boxes1 = [
        [10.0, 10.0, 50.0, 50.0],
        [100.0, 100.0, 150.0, 150.0],
    ]
    boxes2 = [
        [10.0, 10.0, 50.0, 50.0],
        [30.0, 30.0, 70.0, 70.0],
        [200.0, 200.0, 300.0, 300.0],
    ]
    iou_mat = compute_iou_matrix(boxes1, boxes2)
    assert iou_mat.shape == (2, 3)

    for i in range(len(boxes1)):
        for j in range(len(boxes2)):
            expected = compute_iou(boxes1[i], boxes2[j])
            assert np.isclose(iou_mat[i, j], expected, atol=1e-5)


def test_face_tracker_prunes_lost_tracklets_ttl():
    """Verify lost tracklets exceeding max_lost_age are pruned."""
    tracker = FaceTracker(
        iou_threshold=0.3,
        max_lost_frames=2,
        min_hits_to_activate=1,
        max_lost_age=10,
    )
    # Frame 0: detection
    tracker.update([DetectionItem(frame_idx=0, bbox=[10, 10, 50, 50], score=0.9)], frame_idx=0)
    assert 1 in tracker.active_tracklets

    # Frame 1, 2, 3: no detection -> moved to lost_tracklets after max_lost_frames=2
    tracker.update([], frame_idx=1)
    tracker.update([], frame_idx=2)
    tracker.update([], frame_idx=3)
    assert 1 not in tracker.active_tracklets
    assert 1 in tracker.lost_tracklets

    # Frame 15: frame_idx (15) - last_frame (0) > max_lost_age (10) -> pruned
    tracker.update([], frame_idx=15)
    assert 1 not in tracker.lost_tracklets


def test_face_tracker_prunes_lost_tracklets_capacity():
    """Verify lost tracklets exceeding max_lost_tracklets are pruned FIFO."""
    tracker = FaceTracker(
        iou_threshold=0.3,
        max_lost_frames=1,
        min_hits_to_activate=1,
        max_lost_tracklets=3,
        max_lost_age=1000,
    )
    # Create 5 distinct tracklets across separate locations and drop them
    for i in range(5):
        box = [i * 100.0, i * 100.0, i * 100.0 + 40.0, i * 100.0 + 40.0]
        tracker.update([DetectionItem(frame_idx=i, bbox=box, score=0.9)], frame_idx=i)
        # Drop it on next 2 frames
        tracker.update([], frame_idx=i + 1)
        tracker.update([], frame_idx=i + 2)

    # Must be bounded by max_lost_tracklets = 3
    assert len(tracker.lost_tracklets) <= 3


def test_temporal_voting_tie_breaking():
    """Verify deterministic tie-breaking based on higher mean similarity."""
    engine = TemporalVotingEngine(window_size=4, min_consensus_ratio=0.50, min_frames=4)
    tr = Tracklet(track_id=1, start_frame=0, last_frame=0, bbox=[10, 10, 50, 50])

    # 2 votes for person_A with lower similarities (avg 0.65)
    tr.add_detection(DetectionItem(frame_idx=0, bbox=[10, 10, 50, 50], score=0.9, predicted_id="person_A", similarity=0.64))
    tr.add_detection(DetectionItem(frame_idx=1, bbox=[10, 10, 50, 50], score=0.9, predicted_id="person_A", similarity=0.66))

    # 2 votes for person_B with higher similarities (avg 0.85)
    tr.add_detection(DetectionItem(frame_idx=2, bbox=[10, 10, 50, 50], score=0.9, predicted_id="person_B", similarity=0.84))
    tr.add_detection(DetectionItem(frame_idx=3, bbox=[10, 10, 50, 50], score=0.9, predicted_id="person_B", similarity=0.86))

    res = engine.vote(tr)
    # Tie broken in favor of person_B due to higher mean similarity
    assert res.status == "CONFIRMED_MATCH"
    assert res.person_id == "person_B"
    assert np.isclose(res.mean_similarity, 0.85, atol=1e-3)


def test_temporal_voting_excessive_unknown_rejected():
    """Verify that when unknown votes meet or exceed max_unknown_ratio, status is AMBIGUOUS."""
    engine = TemporalVotingEngine(
        window_size=5,
        min_consensus_ratio=0.55,
        min_frames=5,
        max_unknown_ratio=0.40,
    )
    # Case 1: 4 person_A (80%), 1 UNKNOWN (20% < 40%) -> CONFIRMED_MATCH
    tr_clean = Tracklet(track_id=1, start_frame=0, last_frame=0, bbox=[10, 10, 50, 50])
    for i in range(4):
        tr_clean.add_detection(DetectionItem(frame_idx=i, bbox=[10, 10, 50, 50], score=0.9, predicted_id="person_A", similarity=0.8))
    tr_clean.add_detection(DetectionItem(frame_idx=4, bbox=[10, 10, 50, 50], score=0.9, predicted_id="UNKNOWN", similarity=0.0))

    res1 = engine.vote(tr_clean)
    assert res1.status == "CONFIRMED_MATCH"
    assert res1.person_id == "person_A"

    # Case 2: 3 person_A (60%), 2 UNKNOWN (40% >= 40% threshold) -> AMBIGUOUS
    tr_noisy = Tracklet(track_id=2, start_frame=0, last_frame=0, bbox=[10, 10, 50, 50])
    for i in range(3):
        tr_noisy.add_detection(DetectionItem(frame_idx=i, bbox=[10, 10, 50, 50], score=0.9, predicted_id="person_A", similarity=0.8))
    for i in range(3, 5):
        tr_noisy.add_detection(DetectionItem(frame_idx=i, bbox=[10, 10, 50, 50], score=0.9, predicted_id="UNKNOWN", similarity=0.0))

    res2 = engine.vote(tr_noisy)
    assert res2.status == "AMBIGUOUS"
    assert res2.person_id is None


def test_voting_engine_config_validation():
    """Verify TemporalVotingEngine rejects invalid parameters."""
    with pytest.raises(ValueError, match="window_size must be >= 1"):
        TemporalVotingEngine(window_size=0)

    with pytest.raises(ValueError, match="min_frames must be between 1 and window_size"):
        TemporalVotingEngine(window_size=5, min_frames=0)

    with pytest.raises(ValueError, match="min_frames must be between 1 and window_size"):
        TemporalVotingEngine(window_size=5, min_frames=6)

    with pytest.raises(ValueError, match="min_consensus_ratio must be in"):
        TemporalVotingEngine(min_consensus_ratio=-0.1)

    with pytest.raises(ValueError, match="max_unknown_ratio must be in"):
        TemporalVotingEngine(max_unknown_ratio=1.5)

    with pytest.raises(ValueError, match="unknown_label must be non-empty"):
        TemporalVotingEngine(unknown_label="")


def test_face_tracker_config_validation():
    """Verify FaceTracker rejects invalid parameters."""
    with pytest.raises(ValueError, match="iou_threshold must be in"):
        FaceTracker(iou_threshold=-0.1)

    with pytest.raises(ValueError, match="max_lost_frames must be >= 0"):
        FaceTracker(max_lost_frames=-1)

    with pytest.raises(ValueError, match="min_hits_to_activate must be >= 1"):
        FaceTracker(min_hits_to_activate=0)

    with pytest.raises(ValueError, match="max_lost_age must be >= 0"):
        FaceTracker(max_lost_age=-1)

    with pytest.raises(ValueError, match="max_lost_tracklets must be >= 0"):
        FaceTracker(max_lost_tracklets=-1)

    with pytest.raises(ValueError, match="appearance_weight must be in"):
        FaceTracker(appearance_weight=1.5)


def test_zero_max_unknown_ratio_behavior():
    """Verify max_unknown_ratio=0.0 allows confirmation when 0 UNKNOWN votes exist."""
    engine = TemporalVotingEngine(
        window_size=3,
        min_consensus_ratio=0.6,
        min_frames=3,
        max_unknown_ratio=0.0,
    )

    # 3 frames of person_A, 0 UNKNOWN -> CONFIRMED_MATCH
    tr_clean = Tracklet(track_id=1, start_frame=0, last_frame=0, bbox=[10, 10, 50, 50])
    for i in range(3):
        tr_clean.add_detection(DetectionItem(frame_idx=i, bbox=[10, 10, 50, 50], predicted_id="person_A", similarity=0.8))
    res_clean = engine.vote(tr_clean)
    assert res_clean.status == "CONFIRMED_MATCH"
    assert res_clean.person_id == "person_A"

    # 2 frames of person_A, 1 UNKNOWN -> AMBIGUOUS because unknown_votes > 0 and 1/3 >= 0.0
    tr_noisy = Tracklet(track_id=2, start_frame=0, last_frame=0, bbox=[10, 10, 50, 50])
    tr_noisy.add_detection(DetectionItem(frame_idx=0, bbox=[10, 10, 50, 50], predicted_id="person_A", similarity=0.8))
    tr_noisy.add_detection(DetectionItem(frame_idx=1, bbox=[10, 10, 50, 50], predicted_id="person_A", similarity=0.8))
    tr_noisy.add_detection(DetectionItem(frame_idx=2, bbox=[10, 10, 50, 50], predicted_id="UNKNOWN", similarity=0.0))
    res_noisy = engine.vote(tr_noisy)
    assert res_noisy.status == "AMBIGUOUS"


def test_compute_iou_matrix_degenerate_boxes():
    """Verify degenerate or inverted boxes do not trigger division warnings or NaNs."""
    boxes1 = [[10.0, 10.0, 10.0, 10.0], [50.0, 50.0, 40.0, 40.0]]  # zero area and inverted
    boxes2 = [[10.0, 10.0, 50.0, 50.0]]
    iou_mat = compute_iou_matrix(boxes1, boxes2)
    assert iou_mat.shape == (2, 1)
    assert np.all(iou_mat == 0.0)
    assert not np.isnan(iou_mat).any()


def test_temporal_voting_lexical_tie_break_and_nan():
    """Verify tie-break uses person_id lexical order when votes and similarity are equal, and handles NaN."""
    engine = TemporalVotingEngine(window_size=2, min_consensus_ratio=0.5, min_frames=2)
    tr = Tracklet(track_id=1, start_frame=0, last_frame=0, bbox=[10, 10, 50, 50])

    # 1 vote for 'user_z' with NaN similarity (converted to 0.0)
    tr.add_detection(DetectionItem(frame_idx=0, bbox=[10, 10, 50, 50], predicted_id="user_z", similarity=float("nan")))
    # 1 vote for 'user_a' with 0.0 similarity
    tr.add_detection(DetectionItem(frame_idx=1, bbox=[10, 10, 50, 50], predicted_id="user_a", similarity=0.0))

    res = engine.vote(tr)
    # Both have 1 vote and 0.0 similarity; 'user_a' wins lexically over 'user_z'
    assert res.status == "CONFIRMED_MATCH"
    assert res.person_id == "user_a"


def test_frame_delta_semantics():
    """Verify that jumping frame indices (frame skipping) correctly sets time_since_update."""
    tracker = FaceTracker(iou_threshold=0.3, max_lost_frames=10, min_hits_to_activate=1)

    # Frame 100: detected
    tracker.update([DetectionItem(frame_idx=100, bbox=[10, 10, 50, 50])], frame_idx=100)
    assert 1 in tracker.active_tracklets
    assert tracker.active_tracklets[1].last_frame == 100
    assert tracker.active_tracklets[1].time_since_update == 0

    # Skip 5 frames -> frame 105 with no detection: elapsed = 5
    tracker.update([], frame_idx=105)
    assert 1 in tracker.active_tracklets
    assert tracker.active_tracklets[1].time_since_update == 5

    # Skip to frame 115 (> max_lost_frames of 10) -> moves to lost_tracklets
    tracker.update([], frame_idx=115)
    assert 1 not in tracker.active_tracklets
    assert 1 in tracker.lost_tracklets
    assert tracker.lost_tracklets[1].time_since_update == 15


def test_appearance_gated_association():
    """Verify appearance gating prevents ID-switching when two tracks overlap spatially."""
    tracker = FaceTracker(
        iou_threshold=0.2,
        min_hits_to_activate=1,
        appearance_weight=0.5,
        min_appearance_sim=0.5,
    )

    # Unit embeddings
    emb_alice = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    emb_bob = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    # Frame 0: Alice at [10, 10, 50, 50], Bob at [20, 20, 60, 60] (significant IoU)
    det_alice_0 = DetectionItem(frame_idx=0, bbox=[10, 10, 50, 50], embedding=emb_alice)
    det_bob_0 = DetectionItem(frame_idx=0, bbox=[20, 20, 60, 60], embedding=emb_bob)
    active0 = tracker.update([det_alice_0, det_bob_0], frame_idx=0)
    assert len(active0) == 2
    alice_id = active0[0].track_id
    bob_id = active0[1].track_id

    # Frame 1: Swap positions slightly, but retain distinct embeddings
    det_bob_1 = DetectionItem(frame_idx=1, bbox=[15, 15, 55, 55], embedding=emb_bob)
    det_alice_1 = DetectionItem(frame_idx=1, bbox=[12, 12, 52, 52], embedding=emb_alice)
    active1 = tracker.update([det_bob_1, det_alice_1], frame_idx=1)
    track_map = {tr.track_id: tr for tr in active1}

    # Verify Bob matched with Bob's track and Alice with Alice's track
    alice_emb = track_map[alice_id].appearance_embedding
    bob_emb = track_map[bob_id].appearance_embedding
    assert alice_emb is not None
    assert bob_emb is not None
    assert np.dot(alice_emb, emb_alice) > 0.99
    assert np.dot(bob_emb, emb_bob) > 0.99


def test_identity_persistence_across_blinks():
    """Verify tracklet retains confirmed identity during momentary blinks / low-quality frames."""
    engine = TemporalVotingEngine(window_size=3, min_consensus_ratio=0.60, min_frames=2)
    tr = Tracklet(track_id=1, start_frame=0, last_frame=0, bbox=[10, 10, 50, 50])

    # Frame 0 & 1: Good frames -> confirmed match
    tr.add_detection(DetectionItem(frame_idx=0, bbox=[10, 10, 50, 50], predicted_id="Alice", similarity=0.85))
    tr.add_detection(DetectionItem(frame_idx=1, bbox=[10, 10, 50, 50], predicted_id="Alice", similarity=0.86))
    res = engine.vote(tr)
    assert res.status == "CONFIRMED_MATCH"
    assert tr.confirmed_id == "Alice"
    assert tr.is_identity_persistent(current_frame=1, max_hold_frames=10)

    # Frame 2 & 3: Blink occurs -> is_valid_quality is False
    tr.add_detection(DetectionItem(frame_idx=2, bbox=[10, 10, 50, 50], predicted_id="LOW_QUALITY", is_valid_quality=False))
    tr.add_detection(DetectionItem(frame_idx=3, bbox=[10, 10, 50, 50], predicted_id="LOW_QUALITY", is_valid_quality=False))

    # Even during the blink at frame 3, tracklet maintains its persistent identity
    assert tr.is_identity_persistent(current_frame=3, max_hold_frames=10)
    assert tr.confirmed_id == "Alice"

    # Frame 15: After hold period expired (> 10 frames), persistence ends
    assert not tr.is_identity_persistent(current_frame=15, max_hold_frames=10)



