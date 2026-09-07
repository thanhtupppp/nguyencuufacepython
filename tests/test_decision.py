import pytest

from src.recognition.decision import IdentityCandidate, IdentityDecisionEngine


def test_match_requires_threshold_and_margin() -> None:
    engine = IdentityDecisionEngine(threshold=0.70, margin=0.08)
    result = engine.decide([
        IdentityCandidate("person_001", 0.91),
        IdentityCandidate("person_002", 0.80),
    ])
    assert result.status == "MATCHED"
    assert result.person_id == "person_001"
    assert result.margin == pytest.approx(0.11)


def test_high_score_but_small_margin_is_ambiguous() -> None:
    engine = IdentityDecisionEngine(threshold=0.70, margin=0.08)
    result = engine.decide([
        IdentityCandidate("person_001", 0.91),
        IdentityCandidate("person_002", 0.86),
    ])
    assert result.status == "AMBIGUOUS_MATCH"
    assert result.person_id is None


def test_below_threshold_is_unknown() -> None:
    engine = IdentityDecisionEngine(threshold=0.70, margin=0.08)
    result = engine.decide([
        IdentityCandidate("person_001", 0.69),
        IdentityCandidate("person_002", 0.40),
    ])
    assert result.status == "UNKNOWN"
    assert result.person_id is None


def test_empty_candidates_are_unknown() -> None:
    engine = IdentityDecisionEngine(threshold=0.70, margin=0.08)
    result = engine.decide([])
    assert result.status == "UNKNOWN"
    assert result.person_id is None


def test_invalid_configuration_is_rejected() -> None:
    with pytest.raises(ValueError):
        IdentityDecisionEngine(threshold=1.1)
    with pytest.raises(ValueError):
        IdentityDecisionEngine(threshold=0.7, margin=-0.1)
