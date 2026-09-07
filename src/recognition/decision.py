"""Identity decision logic for 1:N face recognition.

The decision layer is deliberately independent from the embedding model and
vector database. It consumes already normalized similarity scores and applies
both an acceptance threshold and a best-vs-second-best margin.
"""

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class IdentityCandidate:
    person_id: str
    score: float


@dataclass(frozen=True)
class IdentityDecision:
    status: str
    person_id: str | None
    top1_score: float
    top2_score: float
    margin: float


class IdentityDecisionEngine:
    """Apply the safety-first 1:N identity decision rule.

    MATCHED requires both:
      top1_score >= threshold
      top1_score - top2_score >= margin

    If the score is high but candidates are too close, AMBIGUOUS_MATCH is
    returned instead of guessing. This is intentional for false-positive
    resistance.
    """

    def __init__(self, threshold: float, margin: float = 0.0) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        if margin < 0.0:
            raise ValueError("margin must be >= 0")
        self.threshold = float(threshold)
        self.margin = float(margin)

    def decide(self, candidates: Iterable[IdentityCandidate]) -> IdentityDecision:
        ranked = sorted(candidates, key=lambda c: c.score, reverse=True)
        if not ranked:
            return IdentityDecision("UNKNOWN", None, 0.0, 0.0, 0.0)

        top1 = ranked[0]
        top2_score = ranked[1].score if len(ranked) > 1 else -1.0
        gap = top1.score - top2_score if len(ranked) > 1 else 1.0

        if top1.score < self.threshold:
            return IdentityDecision("UNKNOWN", None, top1.score, top2_score, gap)
        if gap < self.margin:
            return IdentityDecision("AMBIGUOUS_MATCH", None, top1.score, top2_score, gap)
        return IdentityDecision("MATCHED", top1.person_id, top1.score, top2_score, gap)
