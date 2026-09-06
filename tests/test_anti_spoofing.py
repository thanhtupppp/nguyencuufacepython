"""
Unit tests for Anti-Spoofing and Liveness Detection.
"""

import cv2
import numpy as np
import pytest
from src.anti_spoofing.liveness import (
    AntiSpoofDetector,
    crop_face_with_scale,
    compute_fourier_frequency_score,
    LivenessResult,
)


def test_crop_face_with_scale():
    """Verify multi-scale face cropping with boundary padding."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    bbox = [20.0, 20.0, 80.0, 80.0]

    # Tight crop (scale 1.0)
    crop1 = crop_face_with_scale(img, bbox, scale=1.0, output_size=(80, 80))
    assert crop1.shape == (80, 80, 3)

    # Wide context crop (scale 2.7) - extends beyond 100x100 and must be padded with black borders
    crop2 = crop_face_with_scale(img, bbox, scale=2.7, output_size=(80, 80))
    assert crop2.shape == (80, 80, 3)


def test_fourier_frequency_score():
    """Verify Fourier frequency score computation."""
    # Natural-like gradient face
    natural_face = np.zeros((64, 64), dtype=np.uint8)
    for r in range(64):
        for c in range(64):
            natural_face[r, c] = int(128 + 40 * np.sin(r / 8.0) * np.cos(c / 8.0))

    score_nat = compute_fourier_frequency_score(natural_face)
    assert score_nat >= 0.6

    # Extreme checkerboard / Moiré pattern (sharp artificial high frequencies)
    checkerboard = np.indices((64, 64)).sum(axis=0) % 2 * 255
    score_moire = compute_fourier_frequency_score(checkerboard.astype(np.uint8))
    # Extreme high frequency energy should lower the naturalness score
    assert score_moire <= 0.6


def test_anti_spoof_detector_heuristic_mode():
    """Verify detector runs without error in fallback mode and produces structured result."""
    detector = AntiSpoofDetector(model_path=None, threshold=0.7)

    # Clean gradient face image
    img = np.zeros((150, 150, 3), dtype=np.uint8)
    for r in range(150):
        for c in range(150):
            val = int(120 + 30 * np.sin(r / 10.0))
            img[r, c] = [val, val, val]

    bbox = [25.0, 25.0, 125.0, 125.0]

    res = detector.predict_liveness(img, bbox)
    assert isinstance(res, LivenessResult)
    assert 0.0 <= res.liveness_score <= 1.0
    assert "fourier_heuristic" in res.scale_scores
