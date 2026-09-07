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
    crop1 = crop_face_with_scale(img, bbox, scale=1.0, output_size=(80, 80))
    assert crop1.shape == (80, 80, 3)
    crop2 = crop_face_with_scale(img, bbox, scale=2.7, output_size=(80, 80))
    assert crop2.shape == (80, 80, 3)


def test_fourier_frequency_score():
    """Verify Fourier frequency score computation."""
    natural_face = np.zeros((64, 64), dtype=np.uint8)
    for r in range(64):
        for c in range(64):
            natural_face[r, c] = int(128 + 40 * np.sin(r / 8.0) * np.cos(c / 8.0))
    score_nat = compute_fourier_frequency_score(natural_face)
    assert score_nat >= 0.6

    checkerboard = np.indices((64, 64)).sum(axis=0) % 2 * 255
    score_moire = compute_fourier_frequency_score(checkerboard.astype(np.uint8))
    assert score_moire <= 0.6


def test_anti_spoof_detector_heuristic_mode():
    """Heuristic mode is explicitly opt-in for offline experiments only."""
    detector = AntiSpoofDetector(model_path=None, threshold=0.7, strict_mode=False)
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
    assert res.decision in {LivenessDecision.PASS, LivenessDecision.FAIL, LivenessDecision.INCONCLUSIVE}


def test_anti_spoof_defaults_to_fail_closed_without_model():
    """Production default must not authorize liveness without validated weights."""
    detector = AntiSpoofDetector(model_path=None)
    img = np.full((150, 150, 3), 128, dtype=np.uint8)
    bbox = [20.0, 20.0, 120.0, 120.0]
    res = detector.predict_liveness(img, bbox)
    assert detector.strict_mode is True
    assert res.decision == LivenessDecision.INCONCLUSIVE
    assert res.reason == "MODEL_WEIGHTS_MISSING"
    assert res.is_live is False


def test_anti_spoof_inconclusive_face_too_small():
    """Verify face bbox below min_face_size returns INCONCLUSIVE."""
    detector = AntiSpoofDetector(model_path=None, min_face_size=60)
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    bbox = [10.0, 10.0, 50.0, 50.0]
    res = detector.predict_liveness(img, bbox)
    assert res.decision == LivenessDecision.INCONCLUSIVE
    assert res.reason == "FACE_TOO_SMALL"
    assert res.is_live is False


def test_anti_spoof_inconclusive_extreme_illumination():
    """Verify too dark or washed out image returns INCONCLUSIVE."""
    detector = AntiSpoofDetector(model_path=None)
    dark_img = np.full((150, 150, 3), 10, dtype=np.uint8)
    bbox = [20.0, 20.0, 120.0, 120.0]
    res_dark = detector.predict_liveness(dark_img, bbox)
    assert res_dark.decision == LivenessDecision.INCONCLUSIVE
    assert res_dark.reason == "EXTREME_ILLUMINATION"
    assert res_dark.is_live is False

    bright_img = np.full((150, 150, 3), 245, dtype=np.uint8)
    res_bright = detector.predict_liveness(bright_img, bbox)
    assert res_bright.decision == LivenessDecision.INCONCLUSIVE
    assert res_bright.reason == "EXTREME_ILLUMINATION"
    assert res_bright.is_live is False


def test_anti_spoof_strict_mode_when_weights_missing():
    """Verify strict mode rejects with INCONCLUSIVE if ONNX weights are missing."""
    detector = AntiSpoofDetector(model_path=None, strict_mode=True)
    img = np.full((150, 150, 3), 128, dtype=np.uint8)
    bbox = [20.0, 20.0, 120.0, 120.0]
    res = detector.predict_liveness(img, bbox)
    assert res.decision == LivenessDecision.INCONCLUSIVE
    assert res.reason == "MODEL_WEIGHTS_MISSING"
    assert res.is_live is False
