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
    LivenessDecision,
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
    assert res.decision in {LivenessDecision.PASS, LivenessDecision.FAIL}


def test_anti_spoof_inconclusive_face_too_small():
    """Verify face bbox below min_face_size returns INCONCLUSIVE."""
    detector = AntiSpoofDetector(model_path=None, min_face_size=60)
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    # 40x40 bbox is < 60
    bbox = [10.0, 10.0, 50.0, 50.0]
    res = detector.predict_liveness(img, bbox)
    assert res.decision == LivenessDecision.INCONCLUSIVE
    assert res.reason == "FACE_TOO_SMALL"
    assert res.is_live is False


def test_anti_spoof_inconclusive_extreme_illumination():
    """Verify too dark or washed out face returns INCONCLUSIVE."""
    detector = AntiSpoofDetector(model_path=None)
    # Very dark image (mean brightness < 25)
    dark_img = np.full((150, 150, 3), 10, dtype=np.uint8)
    bbox = [20.0, 20.0, 120.0, 120.0]
    res_dark = detector.predict_liveness(dark_img, bbox)
    assert res_dark.decision == LivenessDecision.INCONCLUSIVE
    assert res_dark.reason == "EXTREME_ILLUMINATION"
    assert res_dark.is_live is False

    # Very bright / washed out image (mean brightness > 235)
    bright_img = np.full((150, 150, 3), 245, dtype=np.uint8)
    res_bright = detector.predict_liveness(bright_img, bbox)
    assert res_bright.decision == LivenessDecision.INCONCLUSIVE
    assert res_bright.reason == "EXTREME_ILLUMINATION"
    assert res_bright.is_live is False


def test_anti_spoof_strict_mode_when_weights_missing():
    """Verify strict_mode=True rejects with INCONCLUSIVE if ONNX weights are missing."""
    detector = AntiSpoofDetector(model_path=None, strict_mode=True)
    img = np.full((150, 150, 3), 128, dtype=np.uint8)
    bbox = [20.0, 20.0, 120.0, 120.0]
    res = detector.predict_liveness(img, bbox)
    assert res.decision == LivenessDecision.INCONCLUSIVE
    assert res.reason == "MODEL_WEIGHTS_MISSING"
    assert res.is_live is False

