"""
Multi-Object Multi-Frame Face Tracking and Temporal Voting Engine.
Associates face detections across video frames to form Tracklets,
selects the best-quality frame within each tracklet, and performs
temporal consensus voting over a sliding window to eliminate spurious false matches.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment


def compute_iou(box1: list[float] | np.ndarray, box2: list[float] | np.ndarray) -> float:
    """Computes Intersection over Union (IoU) between two [x1, y1, x2, y2] bboxes."""
    x1 = max(float(box1[0]), float(box2[0])) if not isinstance(box1[0], float) else max(box1[0], box2[0])
    y1 = max(float(box1[1]), float(box2[1])) if not isinstance(box1[1], float) else max(box1[1], box2[1])
    x2 = min(float(box1[2]), float(box2[2])) if not isinstance(box1[2], float) else min(box1[2], box2[2])
    y2 = min(float(box1[3]), float(box2[3])) if not isinstance(box1[3], float) else min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    b1_w = max(0.0, float(box1[2] - box1[0]))
    b1_h = max(0.0, float(box1[3] - box1[1]))
    area1 = b1_w * b1_h

    b2_w = max(0.0, float(box2[2] - box2[0]))
    b2_h = max(0.0, float(box2[3] - box2[1]))
    area2 = b2_w * b2_h

    union_area = area1 + area2 - inter_area
    if union_area <= 1e-6:
        return 0.0
    return float(inter_area / union_area)


def compute_iou_matrix(boxes1: list[list[float]], boxes2: list[list[float]]) -> np.ndarray:
    """Computes pairwise IoU matrix of shape (len(boxes1), len(boxes2)) via NumPy vectorization.

    Uses np.divide with condition mask to safely handle zero-area boxes without
    emitting runtime division warnings.
    """
    n1, n2 = len(boxes1), len(boxes2)
    if n1 == 0 or n2 == 0:
        return np.zeros((n1, n2), dtype=np.float32)

    b1 = np.asarray(boxes1, dtype=np.float32)[:, None, :]  # (n1, 1, 4)
    b2 = np.asarray(boxes2, dtype=np.float32)[None, :, :]  # (1, n2, 4)

    x1 = np.maximum(b1[..., 0], b2[..., 0])
    y1 = np.maximum(b1[..., 1], b2[..., 1])
    x2 = np.minimum(b1[..., 2], b2[..., 2])
    y2 = np.minimum(b1[..., 3], b2[..., 3])

    inter_area = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area1 = np.maximum(0.0, b1[..., 2] - b1[..., 0]) * np.maximum(0.0, b1[..., 3] - b1[..., 1])
    area2 = np.maximum(0.0, b2[..., 2] - b2[..., 0]) * np.maximum(0.0, b2[..., 3] - b2[..., 1])

    union_area = area1 + area2 - inter_area
    iou = np.zeros_like(union_area, dtype=np.float32)
    np.divide(
        inter_area,
        union_area,
        out=iou,
        where=union_area > 1e-6,
    )
    return iou


@dataclass
class DetectionItem:
    """A face detection observation in a single frame."""

    frame_idx: int
    bbox: list[float]  # [x1, y1, x2, y2]
    score: float = 1.0  # Detection confidence
    landmarks: Optional[np.ndarray] = None  # (5, 2)
    quality_score: float = 1.0  # Quality Gate score [0.0, 1.0]
    aligned_face: Optional[np.ndarray] = None  # 112x112 aligned image
    embedding: Optional[np.ndarray] = None  # 512D vector
    predicted_id: Optional[str] = None  # person_id
    similarity: float = 0.0  # Similarity to best gallery candidate
    margin: float = 0.0  # Top1 - Top2 score gap
    is_valid_quality: bool = True  # Passed Quality Gate
    rejection_reasons: list[str] = field(default_factory=list)

    @property
    def person_id(self) -> Optional[str]:
        return self.predicted_id

    @person_id.setter
    def person_id(self, val: Optional[str]) -> None:
        self.predicted_id = val


@dataclass
class Tracklet:
    """Represents a continuous trajectory of a face across video frames."""

    track_id: int
    start_frame: int
    last_frame: int
    bbox: list[float]
    hits: int = 0
    time_since_update: int = 0
    history: list[DetectionItem] = field(default_factory=list)
    max_history_len: int = 30
    best_quality_score: float = -1.0
    best_item: Optional[DetectionItem] = None
    appearance_embedding: Optional[np.ndarray] = None
    embedding_momentum: float = 0.85

    # Identity persistence (Hysteresis smoothing across eye blinks & momentary frame dips)
    confirmed_id: Optional[str] = None
    confirmed_sim: float = 0.0
    confirmed_consensus: float = 0.0
    last_confirmed_frame: int = -1
    confirmation_hold_frames: int = 30

    def is_identity_persistent(self, current_frame: int, max_hold_frames: Optional[int] = None) -> bool:
        """Returns True if this tracklet was recently confirmed and is within the hold period."""
        if self.confirmed_id is None or self.last_confirmed_frame < 0:
            return False
        hold = max_hold_frames if max_hold_frames is not None else self.confirmation_hold_frames
        return (current_frame - self.last_confirmed_frame) <= hold

    def mark_missed(self, frame_idx: int) -> None:
        """Updates elapsed frames since last successful detection."""
        self.time_since_update = max(0, frame_idx - self.last_frame)

    def add_detection(self, item: DetectionItem) -> None:
        """Appends detection observation, updates best-frame and appearance EMA."""
        self.history.append(item)
        if len(self.history) > self.max_history_len:
            self.history.pop(0)

        self.last_frame = item.frame_idx
        self.bbox = list(item.bbox)
        self.hits += 1
        self.time_since_update = 0

        if item.quality_score > self.best_quality_score:
            self.best_quality_score = item.quality_score
            self.best_item = item

        # Update appearance embedding via L2-normalized EMA
        if item.embedding is not None:
            raw_emb = np.asarray(item.embedding, dtype=np.float32).flatten()
            norm = float(np.linalg.norm(raw_emb))
            if norm > 1e-6:
                emb = raw_emb / norm
                if self.appearance_embedding is None:
                    self.appearance_embedding = emb.copy()
                else:
                    updated = (
                        self.embedding_momentum * self.appearance_embedding
                        + (1.0 - self.embedding_momentum) * emb
                    )
                    up_norm = float(np.linalg.norm(updated))
                    if up_norm > 1e-6:
                        self.appearance_embedding = updated / up_norm


@dataclass
class VotingResult:
    """Consensus voting outcome over a tracklet window."""

    track_id: int
    status: str  # 'CONFIRMED_MATCH', 'AMBIGUOUS', 'UNKNOWN', 'PENDING'
    person_id: Optional[str] = None  # Winning identity if confirmed
    consensus_ratio: float = 0.0  # Ratio of votes for the winner
    mean_similarity: float = 0.0  # Average similarity score among matching votes
    frame_count: int = 0  # Number of evaluated frames
    vote_breakdown: dict[str, int] = field(default_factory=dict)


class TemporalVotingEngine:
    """Evaluates tracklet prediction history over a sliding window

    to produce stable, noise-free identity decisions.
    """

    def __init__(
        self,
        window_size: int = 5,
        min_consensus_ratio: float = 0.60,
        min_frames: int = 3,
        unknown_label: str = "UNKNOWN",
        max_unknown_ratio: float = 0.40,
    ) -> None:
        """Validates and initializes voting engine parameters."""
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        if not 1 <= min_frames <= window_size:
            raise ValueError("min_frames must be between 1 and window_size")
        if not 0.0 <= min_consensus_ratio <= 1.0:
            raise ValueError("min_consensus_ratio must be in [0.0, 1.0]")
        if not 0.0 <= max_unknown_ratio <= 1.0:
            raise ValueError("max_unknown_ratio must be in [0.0, 1.0]")
        if not unknown_label:
            raise ValueError("unknown_label must be non-empty")

        self.window_size = window_size
        self.min_consensus_ratio = min_consensus_ratio
        self.min_frames = min_frames
        self.unknown_label = unknown_label
        self.max_unknown_ratio = max_unknown_ratio

    def vote(self, tracklet: Tracklet) -> VotingResult:
        """Executes consensus voting over the most recent `window_size` frames of a tracklet.

        Deterministic tie-breaking using vote count desc, mean similarity desc, then person_id asc.
        """
        valid_items = [
            item
            for item in tracklet.history[-self.window_size :]
            if item.is_valid_quality and item.predicted_id is not None
        ]

        if len(valid_items) < self.min_frames:
            return VotingResult(
                track_id=tracklet.track_id,
                status="PENDING",
                frame_count=len(valid_items),
                vote_breakdown={},
            )

        vote_counts: dict[str, int] = {}
        sim_scores: dict[str, list[float]] = {}

        for item in valid_items:
            pid = item.predicted_id if item.predicted_id is not None else self.unknown_label
            vote_counts[pid] = vote_counts.get(pid, 0) + 1

            sim = item.similarity
            if not np.isfinite(sim):
                sim = 0.0
            sim_scores.setdefault(pid, []).append(sim)

        total_frames = len(valid_items)
        unknown_votes = vote_counts.get(self.unknown_label, 0)
        unknown_ratio = unknown_votes / total_frames

        # Deterministic sorting: votes count desc, then mean similarity desc, then person_id asc
        sorted_candidates = sorted(
            vote_counts.items(),
            key=lambda item: (
                -item[1],
                -float(np.mean(sim_scores.get(item[0], [0.0]))),
                item[0],
            ),
        )
        winner_id, winner_votes = sorted_candidates[0]
        consensus = winner_votes / total_frames
        winner_sim = float(np.mean(sim_scores.get(winner_id, [0.0])))

        # Check if excessive unknown votes undermine confidence (only if unknown_votes > 0)
        if unknown_votes > 0 and unknown_ratio >= self.max_unknown_ratio:
            return VotingResult(
                track_id=tracklet.track_id,
                status="AMBIGUOUS",
                person_id=None,
                consensus_ratio=consensus,
                mean_similarity=winner_sim,
                frame_count=len(valid_items),
                vote_breakdown=vote_counts,
            )

        if winner_id == self.unknown_label:
            tracklet.confirmed_id = None
            return VotingResult(
                track_id=tracklet.track_id,
                status="UNKNOWN",
                person_id=None,
                consensus_ratio=consensus,
                mean_similarity=winner_sim,
                frame_count=len(valid_items),
                vote_breakdown=vote_counts,
            )

        if consensus >= self.min_consensus_ratio:
            # Update confirmed identity and refresh last confirmed frame
            tracklet.confirmed_id = winner_id
            tracklet.confirmed_sim = winner_sim
            tracklet.confirmed_consensus = consensus
            tracklet.last_confirmed_frame = tracklet.last_frame
            return VotingResult(
                track_id=tracklet.track_id,
                status="CONFIRMED_MATCH",
                person_id=winner_id,
                consensus_ratio=consensus,
                mean_similarity=winner_sim,
                frame_count=len(valid_items),
                vote_breakdown=vote_counts,
            )
        else:
            return VotingResult(
                track_id=tracklet.track_id,
                status="AMBIGUOUS",
                person_id=None,
                consensus_ratio=consensus,
                mean_similarity=winner_sim,
                frame_count=len(valid_items),
                vote_breakdown=vote_counts,
            )


class FaceTracker:
    """Maintains face identity tracks across frames using IoU bipartite matching

    and optional appearance cosine similarity gating.
    Includes automated pruning to avoid memory leaks on continuous streams.
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_lost_frames: int = 15,
        min_hits_to_activate: int = 2,
        store_lost_tracklets: bool = True,
        max_lost_age: int = 300,
        max_lost_tracklets: int = 100,
        appearance_weight: float = 0.25,
        min_appearance_sim: float = 0.45,
        embedding_momentum: float = 0.85,
    ) -> None:
        """Validates and initializes face tracker parameters."""
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be in [0.0, 1.0]")
        if max_lost_frames < 0:
            raise ValueError("max_lost_frames must be >= 0")
        if min_hits_to_activate < 1:
            raise ValueError("min_hits_to_activate must be >= 1")
        if max_lost_age < 0:
            raise ValueError("max_lost_age must be >= 0")
        if max_lost_tracklets < 0:
            raise ValueError("max_lost_tracklets must be >= 0")
        if not 0.0 <= appearance_weight <= 1.0:
            raise ValueError("appearance_weight must be in [0.0, 1.0]")
        if not 0.0 <= min_appearance_sim <= 1.0:
            raise ValueError("min_appearance_sim must be in [0.0, 1.0]")
        if not 0.0 <= embedding_momentum <= 1.0:
            raise ValueError("embedding_momentum must be in [0.0, 1.0]")

        self.iou_threshold = iou_threshold
        self.max_lost_frames = max_lost_frames
        self.min_hits_to_activate = min_hits_to_activate
        self.store_lost_tracklets = store_lost_tracklets
        self.max_lost_age = max_lost_age
        self.max_lost_tracklets = max_lost_tracklets
        self.appearance_weight = appearance_weight
        self.min_appearance_sim = min_appearance_sim
        self.embedding_momentum = embedding_momentum

        self.next_track_id = 1
        self.active_tracklets: dict[int, Tracklet] = {}
        self.lost_tracklets: dict[int, Tracklet] = {}

    def _prune_lost_tracklets(self, current_frame: int) -> None:
        """Evicts stale lost tracklets by TTL (max_lost_age) and capacity (max_lost_tracklets)."""
        if not self.lost_tracklets:
            return

        # 1. TTL-based eviction
        stale_keys = [
            tid
            for tid, tr in self.lost_tracklets.items()
            if (current_frame - tr.last_frame) > self.max_lost_age
        ]
        for tid in stale_keys:
            del self.lost_tracklets[tid]

        # 2. Capacity-based eviction by oldest last_frame
        while len(self.lost_tracklets) > self.max_lost_tracklets:
            oldest_tid = min(
                self.lost_tracklets,
                key=lambda k: self.lost_tracklets[k].last_frame,
            )
            del self.lost_tracklets[oldest_tid]

    def _generate_next_track_id(self) -> int:
        """Generates monotonically increasing track_id."""
        tid = self.next_track_id
        self.next_track_id += 1
        return tid

    def _compute_cost_matrix(
        self,
        tracks: list[Tracklet],
        detections: list[DetectionItem],
        iou_matrix: np.ndarray,
    ) -> np.ndarray:
        """Computes matching cost matrix with IoU and appearance gating."""
        n_tracks = len(tracks)
        n_dets = len(detections)
        large_cost = 1e5
        cost_matrix = np.full((n_tracks, n_dets), large_cost, dtype=np.float32)

        for r, track in enumerate(tracks):
            track_emb = track.appearance_embedding

            for c, det in enumerate(detections):
                iou = float(iou_matrix[r, c])

                # Gate 1: Spatial overlap
                if iou < self.iou_threshold:
                    continue

                det_emb = None
                if det.embedding is not None:
                    raw_emb = np.asarray(det.embedding, dtype=np.float32).flatten()
                    norm = float(np.linalg.norm(raw_emb))
                    if norm > 1e-6:
                        det_emb = raw_emb / norm

                # Gate 2: Appearance cosine similarity (if both embeddings present)
                if (
                    self.appearance_weight > 0.0
                    and track_emb is not None
                    and det_emb is not None
                ):
                    cos_sim = float(np.dot(track_emb, det_emb))
                    if cos_sim < self.min_appearance_sim:
                        continue
                    app_sim = max(0.0, min(1.0, cos_sim))
                    cost = (
                        (1.0 - self.appearance_weight) * (1.0 - iou)
                        + self.appearance_weight * (1.0 - app_sim)
                    )
                else:
                    cost = 1.0 - iou

                cost_matrix[r, c] = cost

        return cost_matrix

    def update(
        self,
        detections: list[DetectionItem],
        frame_idx: int,
    ) -> list[Tracklet]:
        """Updates trackers with detections in current frame.

        :param detections: List of DetectionItem in current frame
        :param frame_idx: Monotonically increasing frame index
        :return: List of currently confirmed/active Tracklets
        """
        if self.store_lost_tracklets:
            self._prune_lost_tracklets(frame_idx)

        tracklet_keys = list(self.active_tracklets.keys())
        active_tracks = [self.active_tracklets[k] for k in tracklet_keys]
        tracklet_boxes = [tr.bbox for tr in active_tracks]
        det_boxes = [d.bbox for d in detections]

        matched_tracks = set()
        matched_dets = set()

        if tracklet_boxes and det_boxes:
            iou_matrix = compute_iou_matrix(tracklet_boxes, det_boxes)
            cost_matrix = self._compute_cost_matrix(
                active_tracks, detections, iou_matrix
            )
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            for r, c in zip(row_ind, col_ind):
                # Discard gated pairs exceeding large cost
                if cost_matrix[r, c] >= 1e4:
                    continue
                tid = tracklet_keys[r]
                self.active_tracklets[tid].add_detection(detections[c])
                matched_tracks.add(tid)
                matched_dets.add(c)

        unmatched_tracks = set(tracklet_keys) - matched_tracks
        for tid in unmatched_tracks:
            tr = self.active_tracklets[tid]
            tr.mark_missed(frame_idx)
            if tr.time_since_update > self.max_lost_frames:
                del self.active_tracklets[tid]
                if self.store_lost_tracklets and self.max_lost_tracklets > 0:
                    self.lost_tracklets[tid] = tr
                    while len(self.lost_tracklets) > self.max_lost_tracklets:
                        oldest_tid = min(
                            self.lost_tracklets,
                            key=lambda k: self.lost_tracklets[k].last_frame,
                        )
                        del self.lost_tracklets[oldest_tid]

        for i, det in enumerate(detections):
            if i not in matched_dets:
                tid = self._generate_next_track_id()
                new_tracklet = Tracklet(
                    track_id=tid,
                    start_frame=frame_idx,
                    last_frame=frame_idx,
                    bbox=list(det.bbox),
                    hits=0,
                    time_since_update=0,
                    embedding_momentum=self.embedding_momentum,
                )
                new_tracklet.add_detection(det)
                self.active_tracklets[tid] = new_tracklet

        return [
            tr
            for tr in self.active_tracklets.values()
            if tr.hits >= self.min_hits_to_activate and tr.time_since_update == 0
        ]
