"""
Tests for benchmark evaluation metrics:
- Cosine similarity
- FAR, FRR, EER calculation
- Top-1 and Margin evaluation
"""

import numpy as np
import pytest
from benchmarks.scripts.metrics import (
    compute_cosine_similarity,
    compute_roc_and_rates,
    evaluate_gallery_probe,
)
from src.recognition.base import BaseFaceRecognizer


def test_l2_normalize():
    """Testing vector normalization."""
    vec = np.array([3.0, 4.0], dtype=np.float32)
    normalized = BaseFaceRecognizer.l2_normalize(vec)
    assert np.isclose(np.linalg.norm(normalized), 1.0)

    # 2D batch normalization
    batch = np.array([[1.0, 2.0, 2.0], [0.0, 5.0, 0.0]], dtype=np.float32)
    norm_batch = BaseFaceRecognizer.l2_normalize(batch)
    norms = np.linalg.norm(norm_batch, axis=1)
    np.testing.assert_allclose(norms, [1.0, 1.0])


def test_cosine_similarity():
    """Testing cosine similarity calculation."""
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([1.0, 0.0, 0.0])
    v3 = np.array([0.0, 1.0, 0.0])

    assert np.isclose(compute_cosine_similarity(v1, v2), 1.0)
    assert np.isclose(compute_cosine_similarity(v1, v3), 0.0)


def test_compute_roc_and_rates():
    """Testing ROC and error rate computation."""
    # Synthetic genuine scores centered around 0.85
    rng = np.random.RandomState(42)
    genuine = rng.normal(loc=0.85, scale=0.05, size=1000)
    # Synthetic impostor scores centered around 0.20
    impostor = rng.normal(loc=0.20, scale=0.08, size=5000)

    results = compute_roc_and_rates(genuine, impostor)
    assert results["EER"] < 0.01  # Clear separation should have EER near 0
    assert 0.3 < results["EER_threshold"] < 0.7
    assert "TAR@FAR=0.001" in results["operating_points"]


def test_evaluate_gallery_probe_with_margin():
    """Testing dual-threshold margin decision logic."""
    gallery = {
        "person_A": [np.array([1.0, 0.0, 0.0])],
        "person_B": [np.array([0.0, 1.0, 0.0])],
    }

    # Probe 1: Clearly person A
    probe_clear_a = {
        "person_id": "person_A",
        "condition": "frontal",
        "embedding": np.array([0.99, 0.05, 0.0]),
    }
    # Normalize probe embedding
    probe_clear_a["embedding"] /= np.linalg.norm(probe_clear_a["embedding"])

    res = evaluate_gallery_probe(gallery, [probe_clear_a], threshold=0.6, margin=0.1)
    assert res["correct_matches"] == 1
    assert res["ambiguous_rejects"] == 0

    # Probe 2: Ambiguous between A and B (similar distance to both)
    probe_ambiguous = {
        "person_id": "person_A",
        "condition": "blur",
        "embedding": np.array([0.707, 0.707, 0.0]),  # 45 degrees between A and B
    }
    res_amb = evaluate_gallery_probe(gallery, [probe_ambiguous], threshold=0.6, margin=0.1)
    # S1 = 0.707, S2 = 0.707 -> margin = 0.0 < 0.1 -> Ambiguous reject!
    assert res_amb["ambiguous_rejects"] == 1
    assert res_amb["false_matches"] == 0
