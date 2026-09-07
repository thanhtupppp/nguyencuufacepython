"""
Multi-object multi-frame face tracking and temporal identity fusion.

Associates face detections across frames using geometry plus optional face
embedding appearance similarity, keeps short-lived lost tracks for recovery,
and performs quality/similarity-weighted temporal consensus.
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from scipy.optimize import linear_sum_assignment


def compute_iou(box1: list[float] | np.ndarray, box2: list[float] | np.ndarray) -> float:
    """Computes IoU between two [x1, y1, x2, y2] bounding boxes."""
    x1 = max(float(box1[0]), float(box2[0])); y1 = max(float(box1[1]), float(box2[1]))
    x2 = min(float(box1[2]), float(box2[2])); y2 = min(float(box1[3]), float(box2[3]))
    inter_w = max(0.0, x2 - x1); inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h
    area1 = max(0.0, float(box1[2]) - float(box1[0])) * max(0.0, float(box1[3]) - float(box1[1]))
    area2 = max(0.0, float(box2[2]) - float(box2[0])) * max(0.0, float(box2[3]) - float(box2[1]))
    union_area = area1 + area2 - inter_area
    return float(inter_area / union_area) if union_area > 1e-6 else 0.0


def compute_iou_matrix(boxes1: list[list[float]], boxes2: list[list[float]]) -> np.ndarray:
    n1, n2 = len(boxes1), len(boxes2)
    if n1 == 0 or n2 == 0:
        return np.zeros((n1, n2), dtype=np.float32)
    return np.asarray([[compute_iou(a, b) for b in boxes2] for a in boxes1], dtype=np.float32)


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Cosine similarity with safe handling for invalid/zero vectors."""
    a = np.asarray(vec1, dtype=np.float32).reshape(-1); b = np.asarray(vec2, dtype=np.float32).reshape(-1)
    if a.shape != b.shape or a.size == 0:
        return -1.0
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 1e-8 and nb > 1e-8 else -1.0


@dataclass
class DetectionItem:
    frame_idx: int
    bbox: list[float]
    score: float
    landmarks: Optional[np.ndarray] = None
    quality_score: float = 1.0
    aligned_face: Optional[np.ndarray] = None
    embedding: Optional[np.ndarray] = None
    predicted_id: Optional[str] = None
    similarity: float = 0.0
    margin: float = 0.0
    is_valid_quality: bool = True
    rejection_reasons: list[str] = field(default_factory=list)


@dataclass
class Tracklet:
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

    def add_detection(self, item: DetectionItem) -> None:
        self.history.append(item)
        if len(self.history) > self.max_history_len:
            self.history.pop(0)
        self.last_frame = item.frame_idx; self.bbox = item.bbox
        self.hits += 1; self.time_since_update = 0
        if item.quality_score > self.best_quality_score:
            self.best_quality_score = item.quality_score; self.best_item = item

    @property
    def latest_embedding(self) -> Optional[np.ndarray]:
        for item in reversed(self.history):
            if item.embedding is not None:
                return item.embedding
        return None


@dataclass
class VotingResult:
    track_id: int
    status: str
    person_id: Optional[str] = None
    consensus_ratio: float = 0.0
    mean_similarity: float = 0.0
    frame_count: int = 0
    vote_breakdown: dict[str, int] = field(default_factory=dict)
    weighted_breakdown: dict[str, float] = field(default_factory=dict)


class TemporalVotingEngine:
    """Quality- and similarity-weighted identity consensus over a sliding window."""

    def __init__(self, window_size: int = 5, min_consensus_ratio: float = 0.60,
                 min_frames: int = 3, unknown_label: str = "UNKNOWN", similarity_floor: float = 0.0):
        if window_size < 1 or min_frames < 1 or min_frames > window_size:
            raise ValueError("min_frames must be between 1 and window_size")
        if not 0.0 < min_consensus_ratio <= 1.0:
            raise ValueError("min_consensus_ratio must be in (0, 1]")
        self.window_size = window_size; self.min_consensus_ratio = min_consensus_ratio
        self.min_frames = min_frames; self.unknown_label = unknown_label; self.similarity_floor = similarity_floor

    def _item_weight(self, item: DetectionItem) -> float:
        quality = float(np.clip(item.quality_score, 0.0, 1.0))
        detection = float(np.clip(item.score, 0.0, 1.0))
        sim = float(np.clip((item.similarity - self.similarity_floor) / max(1e-6, 1.0 - self.similarity_floor), 0.0, 1.0))
        return quality * detection * (0.5 + 0.5 * sim)

    def vote(self, tracklet: Tracklet) -> VotingResult:
        valid_items = [i for i in tracklet.history[-self.window_size:] if i.is_valid_quality and i.predicted_id is not None]
        if len(valid_items) < self.min_frames:
            return VotingResult(track_id=tracklet.track_id, status="PENDING", frame_count=len(valid_items))
        counts: dict[str, int] = {}; weighted: dict[str, float] = {}; sims: dict[str, list[float]] = {}
        for item in valid_items:
            pid = item.predicted_id
            counts[pid] = counts.get(pid, 0) + 1
            weighted[pid] = weighted.get(pid, 0.0) + self._item_weight(item)
            sims.setdefault(pid, []).append(item.similarity)
        winner_id, winner_weight = sorted(weighted.items(), key=lambda x: (-x[1], x[0]))[0]
        total = sum(weighted.values()); weighted_consensus = winner_weight / total if total > 1e-9 else 0.0
        raw_consensus = counts[winner_id] / float(len(valid_items))
        if winner_id == self.unknown_label:
            status, person_id = "UNKNOWN", None
        elif raw_consensus >= self.min_consensus_ratio and weighted_consensus >= self.min_consensus_ratio:
            status, person_id = "CONFIRMED_MATCH", winner_id
        else:
            status, person_id = "AMBIGUOUS", None
        return VotingResult(track_id=tracklet.track_id, status=status, person_id=person_id,
                            consensus_ratio=raw_consensus, mean_similarity=float(np.mean(sims[winner_id])),
                            frame_count=len(valid_items), vote_breakdown=counts, weighted_breakdown=weighted)


class FaceTracker:
    """Multi-face tracker using geometry plus optional embedding appearance."""

    def __init__(self, iou_threshold: float = 0.3, max_lost_frames: int = 15,
                 min_hits_to_activate: int = 2, appearance_weight: float = 0.35,
                 appearance_threshold: float = 0.45, recovery_embedding_threshold: float = 0.60):
        if not 0.0 <= appearance_weight <= 1.0:
            raise ValueError("appearance_weight must be in [0, 1]")
        if not -1.0 <= appearance_threshold <= 1.0 or not -1.0 <= recovery_embedding_threshold <= 1.0:
            raise ValueError("appearance thresholds must be in [-1, 1]")
        self.iou_threshold = iou_threshold; self.max_lost_frames = max_lost_frames
        self.min_hits_to_activate = min_hits_to_activate; self.appearance_weight = appearance_weight
        self.appearance_threshold = appearance_threshold; self.recovery_embedding_threshold = recovery_embedding_threshold
        self.next_track_id = 1; self.active_tracklets: dict[int, Tracklet] = {}; self.lost_tracklets: dict[int, Tracklet] = {}

    def _association_score(self, track: Tracklet, det: DetectionItem) -> float:
        iou = compute_iou(track.bbox, det.bbox); emb = track.latest_embedding
        appearance = cosine_similarity(emb, det.embedding) if emb is not None and det.embedding is not None else -1.0
        if appearance >= self.appearance_threshold:
            return (1.0 - self.appearance_weight) * iou + self.appearance_weight * max(0.0, appearance)
        if emb is None and iou >= self.iou_threshold:
            return iou
        return -1.0

    def _match_active(self, detections: list[DetectionItem]) -> tuple[set[int], set[int]]:
        keys = list(self.active_tracklets.keys()); matched_tracks: set[int] = set(); matched_dets: set[int] = set()
        if not keys or not detections:
            return matched_tracks, matched_dets
        scores = np.full((len(keys), len(detections)), -1.0, dtype=np.float32)
        for r, tid in enumerate(keys):
            for c, det in enumerate(detections):
                iou = compute_iou(self.active_tracklets[tid].bbox, det.bbox)
                emb = self.active_tracklets[tid].latest_embedding
                appearance = cosine_similarity(emb, det.embedding) if emb is not None and det.embedding is not None else -1.0
                if iou >= self.iou_threshold and (emb is None or appearance >= self.appearance_threshold):
                    scores[r, c] = self._association_score(self.active_tracklets[tid], det)
                elif appearance >= self.appearance_threshold:
                    scores[r, c] = self.appearance_weight * max(0.0, appearance)
        rows, cols = linear_sum_assignment(-scores)
        for r, c in zip(rows, cols):
            if scores[r, c] < 0.0: continue
            tid = keys[r]; self.active_tracklets[tid].add_detection(detections[c])
            matched_tracks.add(tid); matched_dets.add(c)
        return matched_tracks, matched_dets

    def _recover_lost(self, detections: list[DetectionItem], unmatched_dets: set[int]) -> set[int]:
        recovered: set[int] = set()
        for tid, track in list(self.lost_tracklets.items()):
            if track.time_since_update > self.max_lost_frames:
                del self.lost_tracklets[tid]; continue
            emb = track.latest_embedding
            if emb is None: continue
            best_idx, best_score = None, -1.0
            for idx in unmatched_dets - recovered:
                det_emb = detections[idx].embedding
                if det_emb is None: continue
                score = cosine_similarity(emb, det_emb)
                if score >= self.recovery_embedding_threshold and score > best_score:
                    best_idx, best_score = idx, score
            if best_idx is not None:
                track.add_detection(detections[best_idx]); self.active_tracklets[tid] = track
                del self.lost_tracklets[tid]; recovered.add(best_idx)
        return recovered

    def update(self, detections: list[DetectionItem], frame_idx: int) -> list[Tracklet]:
        matched_tracks, matched_dets = self._match_active(detections)
        for tid in list(self.active_tracklets):
            if tid not in matched_tracks:
                tr = self.active_tracklets.pop(tid); tr.time_since_update = 1
                self.lost_tracklets[tid] = tr

        unmatched_dets = set(range(len(detections))) - matched_dets
        recovered = self._recover_lost(detections, unmatched_dets); unmatched_dets -= recovered
        for tr in self.lost_tracklets.values(): tr.time_since_update += 1

        for i in sorted(unmatched_dets):
            det = detections[i]
            tr = Tracklet(track_id=self.next_track_id, start_frame=frame_idx, last_frame=frame_idx, bbox=det.bbox)
            tr.add_detection(det); self.active_tracklets[self.next_track_id] = tr; self.next_track_id += 1
        return [tr for tr in self.active_tracklets.values() if tr.hits >= self.min_hits_to_activate and tr.time_since_update == 0]
