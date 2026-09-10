"""Temporal identity confirmation state machine.

Tracking identity is deliberately separate from person identity: ``track_id``
is camera-local while ``person_id`` is attached only after repeated evidence.
This module contains no model inference and is deterministic/replayable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class TrackIdentityState(str, Enum):
    UNKNOWN = "unknown"
    CANDIDATE = "candidate"
    CONFIRMED = "confirmed"
    LOST = "lost"


@dataclass(frozen=True)
class IdentityObservation:
    person_id: str | None
    accepted: bool
    score: float | None = None


@dataclass
class ConfirmationConfig:
    min_consecutive: int = 3
    max_gap_frames: int = 2
    min_score: float = 0.0


@dataclass
class TrackConfirmation:
    track_id: str
    config: ConfirmationConfig = field(default_factory=ConfirmationConfig)
    state: TrackIdentityState = TrackIdentityState.UNKNOWN
    person_id: str | None = None
    candidate_person_id: str | None = None
    consecutive_hits: int = 0
    last_seen_frame: int | None = None

    def observe(self, frame_index: int, observation: IdentityObservation) -> TrackIdentityState:
        if self.last_seen_frame is not None and frame_index <= self.last_seen_frame:
            raise ValueError("frame_index must increase monotonically")

        if self.last_seen_frame is not None and frame_index - self.last_seen_frame - 1 > self.config.max_gap_frames:
            self._reset_candidate()
            self.person_id = None
            self.state = TrackIdentityState.LOST

        self.last_seen_frame = frame_index

        valid = (
            observation.accepted
            and observation.person_id is not None
            and (observation.score is None or observation.score >= self.config.min_score)
        )

        if not valid:
            if self.state != TrackIdentityState.CONFIRMED:
                self._reset_candidate()
                self.state = TrackIdentityState.UNKNOWN
            return self.state

        pid = observation.person_id
        if self.state == TrackIdentityState.CONFIRMED:
            if pid == self.person_id:
                return self.state
            # A conflicting identity must not silently replace a confirmed one.
            self._reset_candidate()
            self.state = TrackIdentityState.CONFIRMED
            return self.state

        if pid == self.candidate_person_id:
            self.consecutive_hits += 1
        else:
            self.candidate_person_id = pid
            self.consecutive_hits = 1
            self.state = TrackIdentityState.CANDIDATE

        if self.consecutive_hits >= self.config.min_consecutive:
            self.person_id = pid
            self.candidate_person_id = None
            self.consecutive_hits = 0
            self.state = TrackIdentityState.CONFIRMED

        return self.state

    def expire(self, frame_index: int) -> TrackIdentityState:
        if self.last_seen_frame is None or frame_index <= self.last_seen_frame:
            raise ValueError("frame_index must be greater than last seen frame")
        if frame_index - self.last_seen_frame - 1 > self.config.max_gap_frames:
            self._reset_candidate()
            self.person_id = None
            self.state = TrackIdentityState.LOST
        return self.state

    def _reset_candidate(self) -> None:
        self.candidate_person_id = None
        self.consecutive_hits = 0
