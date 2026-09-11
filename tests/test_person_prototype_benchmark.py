from __future__ import annotations

import numpy as np

from benchmarks.pgvector_person_prototype_benchmark import normalize, prototype_rank


def test_normalize_produces_unit_vector() -> None:
    result = normalize(np.array([3.0, 4.0], dtype=np.float32))
    assert np.allclose(np.linalg.norm(result), 1.0)


def test_prototype_rank_returns_person_ids_only() -> None:
    query = type("Q", (), {"vector": normalize(np.array([1.0, 0.0], dtype=np.float32))})()
    prototypes = (
        ("person_a", normalize(np.array([1.0, 0.0], dtype=np.float32))),
        ("person_b", normalize(np.array([0.0, 1.0], dtype=np.float32))),
    )
    assert prototype_rank(prototypes, query, 1) == ("person_a",)


def test_prototype_rank_has_deterministic_tie_break() -> None:
    query = type("Q", (), {"vector": normalize(np.array([1.0, 0.0], dtype=np.float32))})()
    prototypes = (
        ("person_b", normalize(np.array([1.0, 0.0], dtype=np.float32))),
        ("person_a", normalize(np.array([1.0, 0.0], dtype=np.float32))),
    )
    assert prototype_rank(prototypes, query, 2) == ("person_a", "person_b")
