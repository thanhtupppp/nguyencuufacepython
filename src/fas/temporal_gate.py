"""Fail-closed temporal fusion for passive face anti-spoofing scores.

This module deliberately does not choose a model or threshold. It converts
per-frame FAS observations into a deterministic gate that can be calibrated
later on a locked validation set.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class FasState(str, Enum):
    UNKNOWN = "UNKNOWN"
    CANDIDATE_LIVE = "CANDIDATE_LIVE"
    LIVE_CONFIRMED = "LIVE_CONFIRMED"
    SPOOF = "SPOOF"
    ERROR = "ERROR"


@dataclass(frozen=True)
class FasObservation:
    frame_index: int
    track_id: str
    verified: bool
    live: Optional[bool]
    score: Optional[float]


@dataclass(frozen=True)
class FasDecision:
    state: FasState
    live_evidence: int
    spoof_evidence: int
    track_id: str
    frame_index: int


class TemporalFasGate:
    """Deterministic, fail-closed temporal FAS gate.

    A LIVE decision requires `live_frames` verified consecutive observations.
    Any unverified/error observation breaks the live streak. A spoof result
    immediately blocks the track. Track changes invalidate prior evidence.
    """

    def __init__(self, live_frames: int = 3, spoof_frames: int = 1) -> None:
        if live_frames < 1 or spoof_frames < 1:
            raise ValueError("live_frames and spoof_frames must be >= 1")
        self.live_frames = live_frames
        self.spoof_frames = spoof_frames
        self._track_id: Optional[str] = None
        self._last_frame: Optional[int] = None
        self._live_streak = 0
        self._spoof_streak = 0
        self._state = FasState.UNKNOWN

    def reset(self) -> None:
        self._track_id = None
        self._last_frame = None
        self._live_streak = 0
        self._spoof_streak = 0
        self._state = FasState.UNKNOWN

    def update(self, obs: FasObservation) -> FasDecision:
        if self._track_id != obs.track_id:
            self.reset()
            self._track_id = obs.track_id

        if self._last_frame is not None and obs.frame_index <= self._last_frame:
            raise ValueError("frame_index must increase monotonically")
        self._last_frame = obs.frame_index

        if not obs.verified or obs.live is None:
            self._live_streak = 0
            self._spoof_streak = 0
            self._state = FasState.ERROR if not obs.verified else FasState.UNKNOWN
        elif obs.live:
            self._spoof_streak = 0
            self._live_streak += 1
            self._state = (
                FasState.LIVE_CONFIRMED
                if self._live_streak >= self.live_frames
                else FasState.CANDIDATE_LIVE
            )
        else:
            self._live_streak = 0
            self._spoof_streak += 1
            self._state = (
                FasState.SPOOF
                if self._spoof_streak >= self.spoof_frames
                else FasState.UNKNOWN
            )

        return FasDecision(
            state=self._state,
            live_evidence=self._live_streak,
            spoof_evidence=self._spoof_streak,
            track_id=obs.track_id,
            frame_index=obs.frame_index,
        )
