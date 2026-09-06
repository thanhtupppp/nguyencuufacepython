"""
Multi-Object Multi-Frame Face Tracking and Temporal Voting Engine.
Associates face detections across video frames to form Tracklets,
selects the best-quality frame within each tracklet, and performs
temporal consensus voting over a sliding window to eliminate spurious false matches.
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from scipy.optimize import linear_sum_assignment


def compute_iou(box1: list[float] | np.ndarray, box2: list[float] | np.ndarray) -> float:
    """Computes Intersection over Union (IoU) between two [x1, y1, x2, y2] bboxes."""
    x1 = max(float(box1[0]), float(box2[0]))
    y1 = max(float(box1[1]), float(box2[1]))
    x2 = min(float(box1[2]), float(box2[2]))
    y2 = min(float(box1[3]), float(box2[3]))

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, float(box1[2]) - float(box1[0])) * max(0.0, float(box1[3]) - float(box1[1]))
    area2 = max(0.0, float(box2[2]) - float(box2[0])) * max(0.0, float(box2[3]) - float(box2[1]))

    union_area = area1 + area2 - inter_area
    if union_area <= 1e-6:
        return 0.0
    return float(inter_area / union_area)


def compute_iou_matrix(boxes1: list[list[float]], boxes2: list[list[float]]) -> np.ndarray:
    """Computes pairwise IoU matrix of shape (len(boxes1), len(boxes2))."""
    n1 = len(boxes1)
    n2 = len(boxes2)
    if n1 == 0 or n2 == 0:
        return np.zeros((n1, n2), dtype=np.float32)

    iou_mat = np.zeros((n1, n2), dtype=np.float32)
    for i in range(n1):
        for j in range(n2):
            iou_mat[i, j] = compute_iou(boxes1[i], boxes2[j])
    return iou_mat


@dataclass
class DetectionItem:
    """A face detection observation in a single frame."""
    frame_idx: int
    bbox: list[float]                     # [x1, y1, x2, y2]
    score: float                          # Detection confidence
    landmarks: Optional[np.ndarray] = None # (5, 2)
    quality_score: float = 1.0            # Quality Gate score [0.0, 1.0]
    aligned_face: Optional[np.ndarray] = None # 112x112 aligned image
    embedding: Optional[np.ndarray] = None    # 512D vector
    predicted_id: Optional[str] = None        # person_id
    similarity: float = 0.0                   # Similarity to best gallery candidate
    margin: float = 0.0                       # Top1 - Top2 score gap
    is_valid_quality: bool = True             # Passed Quality Gate


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

    # Best-frame selection cache
    best_quality_score: float = -1.0
    best_item: Optional[DetectionItem] = None

    def add_detection(self, item: DetectionItem) -> None:
        """Appends detection observation and updates best-frame selection."""
        self.history.append(item)
        if len(self.history) > self.max_history_len:
            self.history.pop(0)

        self.last_frame = item.frame_idx
        self.bbox = item.bbox
        self.hits += 1
        self.time_since_update = 0

        # Update best frame if current detection has higher quality score
        if item.quality_score > self.best_quality_score:
            self.best_quality_score = item.quality_score
            self.best_item = item


@dataclass
class VotingResult:
    """Consensus voting outcome over a tracklet window."""
    track_id: int
    status: str                       # 'CONFIRMED_MATCH', 'AMBIGUOUS', 'UNKNOWN', 'PENDING'
    person_id: Optional[str] = None   # Winning identity if confirmed
    consensus_ratio: float = 0.0      # Ratio of votes for the winner
    mean_similarity: float = 0.0      # Average similarity score among matching votes
    frame_count: int = 0              # Number of evaluated frames
    vote_breakdown: dict[str, int] = field(default_factory=dict)


class TemporalVotingEngine:
    """
    Evaluates tracklet prediction history over a sliding window
    to produce stable, noise-free identity decisions.
    """

    def __init__(
        self,
        window_size: int = 5,
        min_consensus_ratio: float = 0.60,
        min_frames: int = 3,
        unknown_label: str = "UNKNOWN",
    ):
        self.window_size = window_size
        self.min_consensus_ratio = min_consensus_ratio
        self.min_frames = min_frames
        self.unknown_label = unknown_label

    def vote(self, tracklet: Tracklet) -> VotingResult:
        """
        Executes consensus voting over the most recent `window_size` frames of a tracklet.
        """
        # Filter for valid observations that have a prediction
        valid_items = [
            item for item in tracklet.history[-self.window_size:]
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
            pid = item.predicted_id
            vote_counts[pid] = vote_counts.get(pid, 0) + 1
            sim_scores.setdefault(pid, []).append(item.similarity)

        # Find top candidate
        sorted_candidates = sorted(vote_counts.items(), key=lambda x: x[1], reverse=True)
        winner_id, winner_votes = sorted_candidates[0]
        consensus = winner_votes / float(len(valid_items))

        # Check if winner is UNKNOWN
        if winner_id == self.unknown_label:
            return VotingResult(
                track_id=tracklet.track_id,
                status="UNKNOWN",
                person_id=None,
                consensus_ratio=consensus,
                mean_similarity=float(np.mean(sim_scores.get(winner_id, [0.0]))),
                frame_count=len(valid_items),
                vote_breakdown=vote_counts,
            )

        # Check consensus threshold
        if consensus >= self.min_consensus_ratio:
            return VotingResult(
                track_id=tracklet.track_id,
                status="CONFIRMED_MATCH",
                person_id=winner_id,
                consensus_ratio=consensus,
                mean_similarity=float(np.mean(sim_scores[winner_id])),
                frame_count=len(valid_items),
                vote_breakdown=vote_counts,
            )
        else:
            # Conflicting split vote -> Ambiguous
            return VotingResult(
                track_id=tracklet.track_id,
                status="AMBIGUOUS",
                person_id=None,
                consensus_ratio=consensus,
                mean_similarity=float(np.mean(sim_scores[winner_id])),
                frame_count=len(valid_items),
                vote_breakdown=vote_counts,
            )


class FaceTracker:
    """
    Maintains face identity tracks across frames using IoU bipartite matching.
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_lost_frames: int = 15,
        min_hits_to_activate: int = 2,
    ):
        self.iou_threshold = iou_threshold
        self.max_lost_frames = max_lost_frames
        self.min_hits_to_activate = min_hits_to_activate

        self.next_track_id = 1
        self.active_tracklets: dict[int, Tracklet] = {}
        self.lost_tracklets: dict[int, Tracklet] = {}

    def update(
        self,
        detections: list[DetectionItem],
        frame_idx: int,
    ) -> list[Tracklet]:
        """
        Updates trackers with detections in current frame.

        :param detections: List of DetectionItem in current frame
        :param frame_idx: Monotonically increasing frame index
        :return: List of currently confirmed/active Tracklets
        """
        tracklet_keys = list(self.active_tracklets.keys())
        tracklet_boxes = [self.active_tracklets[k].bbox for k in tracklet_keys]
        det_boxes = [d.bbox for d in detections]

        matched_tracks = set()
        matched_dets = set()

        if tracklet_boxes and det_boxes:
            iou_matrix = compute_iou_matrix(tracklet_boxes, det_boxes)
            # Maximum weight bipartite matching using Hungarian algorithm
            cost_matrix = 1.0 - iou_matrix
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            for r, c in zip(row_ind, col_ind):
                if iou_matrix[r, c] >= self.iou_threshold:
                    tid = tracklet_keys[r]
                    self.active_tracklets[tid].add_detection(detections[c])
                    matched_tracks.add(tid)
                    matched_dets.add(c)

        # Increment lost time for unmatched active tracks
        unmatched_tracks = set(tracklet_keys) - matched_tracks
        for tid in unmatched_tracks:
            tr = self.active_tracklets[tid]
            tr.time_since_update += 1
            if tr.time_since_update > self.max_lost_frames:
                del self.active_tracklets[tid]
                self.lost_tracklets[tid] = tr

        # Create new tracklets for unmatched detections
        for i, det in enumerate(detections):
            if i not in matched_dets:
                new_tracklet = Tracklet(
                    track_id=self.next_track_id,
                    start_frame=frame_idx,
                    last_frame=frame_idx,
                    bbox=det.bbox,
                    hits=0,
                    time_since_update=0,
                )
                new_tracklet.add_detection(det)
                self.active_tracklets[self.next_track_id] = new_tracklet
                self.next_track_id += 1

        # Return active tracks that have met minimum hits threshold
        return [
            tr for tr in self.active_tracklets.values()
            if tr.hits >= self.min_hits_to_activate and tr.time_since_update == 0
        ]
