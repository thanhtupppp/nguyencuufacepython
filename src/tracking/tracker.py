"""
Multi-object multi-frame face tracking and temporal identity fusion.

Associates face detections across frames using geometry plus optional face
embedding appearance similarity, keeps short-lived lost tracks for recovery,
and performs quality/similarity-weighted temporal consensus.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


def compute_iou(box1: Sequence[float] | np.ndarray, box2: Sequence[float] | np.ndarray) -> float:
    """Computes IoU between two [x1, y1, x2, y2] bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, float(x2 - x1))
    inter_h = max(0.0, float(y2 - y1))
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
    """Computes pairwise IoU matrix of shape (len(boxes1), len(boxes2)) via NumPy vectorization."""
    n1, n2 = len(boxes1), len(boxes2)
    if n1 == 0 or n2 == 0:
        return np.zeros((n1, n2), dtype=np.float32)

    b1 = np.asarray(boxes1, dtype=np.float32)[:, None, :]  # (N1, 1, 4)
    b2 = np.asarray(boxes2, dtype=np.float32)[None, :, :]  # (1, N2, 4)

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


def cosine_similarity(emb1: Optional[np.ndarray], emb2: Optional[np.ndarray]) -> float:
    """Computes cosine similarity between two 1D feature vectors."""
    if emb1 is None or emb2 is None:
        return -1.0
    v1 = np.asarray(emb1, dtype=np.float32).flatten()
    v2 = np.asarray(emb2, dtype=np.float32).flatten()
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 <= 1e-6 or n2 <= 1e-6:
        return -1.0
    return float(np.dot(v1, v2) / (n1 * n2))


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


@dataclass
class Tracklet:
    """Tracks a single face across consecutive video frames."""

    track_id: int
    start_frame: int
    last_frame: int
    bbox: list[float]
    hits: int = 1
    time_since_update: int = 0
    history: list[DetectionItem] = field(default_factory=list)
    max_history: int = 30
    best_quality_score: float = -1.0
    best_item: Optional[DetectionItem] = None
    appearance_embedding: Optional[np.ndarray] = None
    embedding_momentum: float = 0.8

    # Identity persistence / Hysteresis smoothing
    confirmed_id: Optional[str] = None
    confirmed_sim: float = 0.0
    confirmed_consensus: float = 0.0
    last_confirmed_frame: int = -1
    confirmation_hold_frames: int = 30

    @property
    def latest_embedding(self) -> Optional[np.ndarray]:
        """Returns the most recent valid embedding in observation history, fallback to EMA."""
        for item in reversed(self.history):
            if item.embedding is not None:
                return item.embedding
        return self.appearance_embedding

    def mark_missed(self, frame_idx: int) -> None:
        """Increments time_since_update using frame delta semantics."""
        elapsed = frame_idx - self.last_frame
        self.time_since_update = elapsed if elapsed > 0 else (self.time_since_update + 1)

    def add_detection(self, det: DetectionItem) -> None:
        """Updates tracklet state with a newly associated detection."""
        self.bbox = list(det.bbox)
        self.last_frame = det.frame_idx
        self.hits += 1
        self.time_since_update = 0
        self.history.append(det)
        if len(self.history) > self.max_history:
            self.history.pop(0)

        # Update best frame observation based on quality_score
        if det.quality_score > self.best_quality_score:
            self.best_quality_score = det.quality_score
            self.best_item = det

        # Update appearance embedding via Exponential Moving Average (EMA)
        if det.embedding is not None:
            raw_emb = np.asarray(det.embedding, dtype=np.float32).flatten()
            norm = np.linalg.norm(raw_emb)
            if norm > 1e-6:
                norm_emb = raw_emb / norm
                if self.appearance_embedding is None:
                    self.appearance_embedding = norm_emb
                else:
                    updated = (
                        self.embedding_momentum * self.appearance_embedding
                        + (1.0 - self.embedding_momentum) * norm_emb
                    )
                    up_norm = np.linalg.norm(updated)
                    if up_norm > 1e-6:
                        self.appearance_embedding = updated / up_norm

    def set_confirmed_identity(
        self,
        person_id: str,
        similarity: float,
        consensus: float,
        frame_idx: int,
    ) -> None:
        """Records confirmed identity for hysteresis smoothing across blinks."""
        self.confirmed_id = person_id
        self.confirmed_sim = similarity
        self.confirmed_consensus = consensus
        self.last_confirmed_frame = frame_idx

    def is_identity_persistent(
        self,
        current_frame: int,
        max_hold_frames: Optional[int] = None,
    ) -> bool:
        """Returns True if confirmed identity is still within the hold window."""
        if self.confirmed_id is None or self.last_confirmed_frame < 0:
            return False
        hold_limit = max_hold_frames if max_hold_frames is not None else self.confirmation_hold_frames
        return (current_frame - self.last_confirmed_frame) <= hold_limit


@dataclass
class VotingResult:
    """Result of temporal voting for a tracklet."""

    track_id: int
    status: str  # "CONFIRMED_MATCH", "AMBIGUOUS", "UNKNOWN", "PENDING"
    person_id: Optional[str] = None
    consensus_ratio: float = 0.0
    mean_similarity: float = 0.0
    frame_count: int = 0
    vote_breakdown: dict[str, int] = field(default_factory=dict)
    weighted_breakdown: dict[str, float] = field(default_factory=dict)


class TemporalVotingEngine:
    """Quality- and similarity-weighted identity consensus over a sliding window."""

    def __init__(
        self,
        window_size: int = 5,
        min_consensus_ratio: float = 0.60,
        min_frames: int = 3,
        unknown_label: str = "UNKNOWN",
        max_unknown_ratio: float = 0.40,
        similarity_floor: float = 0.0,
    ):
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        if min_frames < 1 or min_frames > window_size:
            raise ValueError("min_frames must be between 1 and window_size")
        if not (0.0 < min_consensus_ratio <= 1.0):
            raise ValueError("min_consensus_ratio must be in (0, 1]")
        if not (0.0 <= max_unknown_ratio <= 1.0):
            raise ValueError("max_unknown_ratio must be in [0.0, 1.0]")
        if not unknown_label or not isinstance(unknown_label, str):
            raise ValueError("unknown_label must be non-empty string")

        self.window_size = window_size
        self.min_consensus_ratio = min_consensus_ratio
        self.min_frames = min_frames
        self.unknown_label = unknown_label
        self.max_unknown_ratio = max_unknown_ratio
        self.similarity_floor = similarity_floor

    def _item_weight(self, item: DetectionItem) -> float:
        """Calculates evidence weight based on quality score, detector confidence, and similarity."""
        quality = float(np.clip(item.quality_score, 0.0, 1.0))
        detection = float(np.clip(item.score, 0.0, 1.0))
        sim_val = item.similarity if np.isfinite(item.similarity) else 0.0
        sim = float(np.clip((sim_val - self.similarity_floor) / max(1e-6, 1.0 - self.similarity_floor), 0.0, 1.0))
        return quality * detection * (0.5 + 0.5 * sim)

    def vote(self, tracklet: Tracklet) -> VotingResult:
        """Aggregates identities across sliding window with weighted temporal fusion."""
        valid_items = [
            item
            for item in tracklet.history[-self.window_size :]
            if item.is_valid_quality
        ]

        if len(valid_items) < self.min_frames:
            return VotingResult(
                track_id=tracklet.track_id,
                status="PENDING",
                person_id=None,
                consensus_ratio=0.0,
                mean_similarity=0.0,
                frame_count=len(valid_items),
                vote_breakdown={},
                weighted_breakdown={},
            )

        counts: dict[str, int] = {}
        weighted: dict[str, float] = {}
        sim_scores: dict[str, list[float]] = {}

        for item in valid_items:
            pid = item.predicted_id if item.predicted_id is not None else self.unknown_label
            counts[pid] = counts.get(pid, 0) + 1
            w = self._item_weight(item)
            weighted[pid] = weighted.get(pid, 0.0) + w

            sim = item.similarity if np.isfinite(item.similarity) else 0.0
            sim_scores.setdefault(pid, []).append(sim)

        total_frames = len(valid_items)
        unknown_votes = counts.get(self.unknown_label, 0)
        unknown_ratio = unknown_votes / float(total_frames)

        # Check unknown ratio threshold
        if unknown_votes > 0 and unknown_ratio >= self.max_unknown_ratio:
            if unknown_ratio >= self.min_consensus_ratio:
                status = "UNKNOWN"
            else:
                status = "AMBIGUOUS"
            return VotingResult(
                track_id=tracklet.track_id,
                status=status,
                person_id=None,
                consensus_ratio=unknown_ratio,
                mean_similarity=float(np.mean(sim_scores.get(self.unknown_label, [0.0]))),
                frame_count=total_frames,
                vote_breakdown=counts,
                weighted_breakdown=weighted,
            )

        # Deterministic sorting: weighted score desc, then mean similarity desc, then person_id lexical asc
        sorted_candidates = sorted(
            weighted.items(),
            key=lambda item: (
                -item[1],
                -float(np.mean(sim_scores.get(item[0], [0.0]))),
                item[0],
            ),
        )

        winner_id, winner_weight = sorted_candidates[0]
        total_weight = sum(weighted.values())
        weighted_consensus = winner_weight / total_weight if total_weight > 1e-9 else 0.0
        raw_votes = counts.get(winner_id, 0)
        raw_consensus = raw_votes / float(total_frames)
        winner_sim = float(np.mean(sim_scores.get(winner_id, [0.0])))

        if winner_id == self.unknown_label:
            status = "UNKNOWN"
            person_id = None
        else:
            can_confirm = (
                raw_consensus >= self.min_consensus_ratio
                or (weighted_consensus >= self.min_consensus_ratio and raw_votes >= 2)
            )

            if can_confirm:
                status = "CONFIRMED_MATCH"
                person_id = winner_id
                tracklet.set_confirmed_identity(
                    person_id=winner_id,
                    similarity=winner_sim,
                    consensus=raw_consensus,
                    frame_idx=tracklet.last_frame,
                )
            else:
                status = "AMBIGUOUS"
                person_id = None

        return VotingResult(
            track_id=tracklet.track_id,
            status=status,
            person_id=person_id,
            consensus_ratio=raw_consensus,
            mean_similarity=winner_sim,
            frame_count=total_frames,
            vote_breakdown=counts,
            weighted_breakdown=weighted,
        )


class FaceTracker:
    """Multi-face tracker using geometry plus optional embedding appearance."""

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_lost_frames: int = 15,
        min_hits_to_activate: int = 2,
        store_lost_tracklets: bool = True,
        max_lost_age: int = 60,
        max_lost_tracklets: int = 100,
        appearance_weight: float = 0.35,
        appearance_threshold: float = 0.45,
        min_appearance_sim: Optional[float] = None,
        recovery_embedding_threshold: float = 0.60,
        embedding_momentum: float = 0.8,
    ):
        if not (0.0 <= iou_threshold <= 1.0):
            raise ValueError("iou_threshold must be in [0.0, 1.0]")
        if max_lost_frames < 0:
            raise ValueError("max_lost_frames must be >= 0")
        if min_hits_to_activate < 1:
            raise ValueError("min_hits_to_activate must be >= 1")
        if max_lost_age < 0:
            raise ValueError("max_lost_age must be >= 0")
        if max_lost_tracklets < 0:
            raise ValueError("max_lost_tracklets must be >= 0")
        if not (0.0 <= appearance_weight <= 1.0):
            raise ValueError("appearance_weight must be in [0.0, 1.0]")

        if min_appearance_sim is not None:
            if not (-1.0 <= min_appearance_sim <= 1.0):
                raise ValueError("min_appearance_sim must be in [-1.0, 1.0]")
            appearance_threshold = min_appearance_sim

        if not (-1.0 <= appearance_threshold <= 1.0):
            raise ValueError("appearance_threshold must be in [-1.0, 1.0]")
        if not (-1.0 <= recovery_embedding_threshold <= 1.0):
            raise ValueError("recovery_embedding_threshold must be in [-1.0, 1.0]")
        if not (0.0 <= embedding_momentum <= 1.0):
            raise ValueError("embedding_momentum must be in [0.0, 1.0]")

        self.iou_threshold = iou_threshold
        self.max_lost_frames = max_lost_frames
        self.min_hits_to_activate = min_hits_to_activate
        self.store_lost_tracklets = store_lost_tracklets
        self.max_lost_age = max_lost_age
        self.max_lost_tracklets = max_lost_tracklets
        self.appearance_weight = appearance_weight
        self.appearance_threshold = appearance_threshold
        self.min_appearance_sim = appearance_threshold
        self.recovery_embedding_threshold = recovery_embedding_threshold
        self.embedding_momentum = embedding_momentum

        self.next_track_id = 1
        self.active_tracklets: dict[int, Tracklet] = {}
        self.lost_tracklets: dict[int, Tracklet] = {}

    def _association_score(self, track: Tracklet, det: DetectionItem) -> float:
        """Computes combined spatial-appearance affinity score (>= 0.0 for match candidate)."""
        iou = compute_iou(track.bbox, det.bbox)
        emb = track.latest_embedding
        app = cosine_similarity(emb, det.embedding) if emb is not None and det.embedding is not None else -1.0

        if app >= self.appearance_threshold and self.appearance_weight > 0.0:
            return (1.0 - self.appearance_weight) * iou + self.appearance_weight * max(0.0, app)

        if app >= 0.0 and app < self.appearance_threshold and self.appearance_weight > 0.0:
            return -1.0

        if iou >= self.iou_threshold:
            return iou
        return -1.0

    def _match_active(self, detections: list[DetectionItem]) -> tuple[set[int], set[int]]:
        """Matches active tracks with current detections using Hungarian assignment."""
        keys = list(self.active_tracklets.keys())
        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()

        if not keys or not detections:
            return matched_tracks, matched_dets

        scores = np.full((len(keys), len(detections)), -1.0, dtype=np.float32)
        for r, tid in enumerate(keys):
            track = self.active_tracklets[tid]
            for c, det in enumerate(detections):
                scores[r, c] = self._association_score(track, det)

        rows, cols = linear_sum_assignment(-scores)
        for r, c in zip(rows, cols):
            if scores[r, c] < 0.0:
                continue
            tid = keys[r]
            self.active_tracklets[tid].add_detection(detections[c])
            if tid in self.lost_tracklets:
                del self.lost_tracklets[tid]
            matched_tracks.add(tid)
            matched_dets.add(c)

        return matched_tracks, matched_dets

    def _recover_lost(
        self,
        detections: list[DetectionItem],
        unmatched_dets: set[int],
    ) -> set[int]:
        """Recovers lost tracks whose embeddings match unmatched detections."""
        recovered: set[int] = set()
        for tid, track in list(self.lost_tracklets.items()):
            if track.time_since_update > self.max_lost_age:
                del self.lost_tracklets[tid]
                continue

            emb = track.latest_embedding
            if emb is None:
                continue

            best_idx = None
            best_score = -1.0
            for idx in unmatched_dets - recovered:
                det_emb = detections[idx].embedding
                if det_emb is None:
                    continue
                score = cosine_similarity(emb, det_emb)
                if score >= self.recovery_embedding_threshold and score > best_score:
                    best_idx = idx
                    best_score = score

            if best_idx is not None:
                track.add_detection(detections[best_idx])
                self.active_tracklets[tid] = track
                del self.lost_tracklets[tid]
                recovered.add(best_idx)

        return recovered

    def _prune_lost_tracklets(self, current_frame: int) -> None:
        """Evicts stale lost tracklets by TTL (max_lost_age) and capacity."""
        if not self.lost_tracklets:
            return

        stale_keys = [
            tid
            for tid, tr in self.lost_tracklets.items()
            if (current_frame - tr.last_frame) > self.max_lost_age
        ]
        for tid in stale_keys:
            del self.lost_tracklets[tid]

        while len(self.lost_tracklets) > self.max_lost_tracklets:
            oldest_tid = min(
                self.lost_tracklets,
                key=lambda k: self.lost_tracklets[k].last_frame,
            )
            del self.lost_tracklets[oldest_tid]

    def update(
        self,
        detections: list[DetectionItem],
        frame_idx: int,
    ) -> list[Tracklet]:
        """Updates trackers with detections in current frame."""
        if self.store_lost_tracklets:
            self._prune_lost_tracklets(frame_idx)

        # 1. Match active tracks
        matched_tracks, matched_dets = self._match_active(detections)

        # 2. Handle unmatched active tracks
        unmatched_active = set(self.active_tracklets.keys()) - matched_tracks
        for tid in unmatched_active:
            tr = self.active_tracklets[tid]
            tr.mark_missed(frame_idx)
            if self.store_lost_tracklets and self.max_lost_tracklets > 0:
                self.lost_tracklets[tid] = tr

            if tr.time_since_update > self.max_lost_frames:
                del self.active_tracklets[tid]

        # 3. Try to recover lost tracks using unmatched detections
        unmatched_dets = set(range(len(detections))) - matched_dets
        recovered_dets = self._recover_lost(detections, unmatched_dets)
        unmatched_dets -= recovered_dets

        # 4. Initialize new tracklets for remaining unmatched detections
        for i in sorted(unmatched_dets):
            det = detections[i]
            tid = self.next_track_id
            self.next_track_id += 1
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
