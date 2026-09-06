"""
Tests for Quality Gate: Blur, Size, Pose, and Illumination checks.
"""

import cv2
import numpy as np
import pytest
from src.quality.quality_gate import FaceQualityGate
from src.alignment.aligner import ARCFACE_REFERENCE_LANDMARKS_112


def test_quality_gate_clean_face():
    """Clean sharp face crop should pass all quality checks."""
    gate = FaceQualityGate(min_face_size=50, blur_threshold=30.0)
    # Generate clean image with sharp edges
    img = np.zeros((150, 150, 3), dtype=np.uint8)
    cv2.circle(img, (75, 75), 50, (180, 180, 180), -1)
    # Add high frequency noise/texture so blur variance is high
    noise = np.random.randint(0, 100, (150, 150, 3), dtype=np.uint8)
    img = cv2.add(img, noise)

    bbox = [10, 10, 140, 140]
    landmarks = ARCFACE_REFERENCE_LANDMARKS_112.copy() + 15.0

    res = gate.assess_quality(img, bbox, landmarks)
    assert res.is_valid is True
    assert len(res.rejection_reasons) == 0
    assert res.overall_quality_score > 0.4


def test_quality_gate_rejects_blurry_face():
    """Heavily blurred face should be rejected by blur threshold."""
    gate = FaceQualityGate(blur_threshold=50.0)
    img = np.ones((150, 150, 3), dtype=np.uint8) * 128
    # Blurred smooth image has zero Laplacian variance
    bbox = [10, 10, 140, 140]

    res = gate.assess_quality(img, bbox)
    assert res.is_valid is False
    assert any("BLURRY_FACE" in r for r in res.rejection_reasons)


def test_quality_gate_rejects_small_face():
    """Bounding box smaller than min_face_size should be rejected."""
    gate = FaceQualityGate(min_face_size=60)
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    bbox = [10, 10, 40, 40]  # Width = 30, Height = 30 < 60

    res = gate.assess_quality(img, bbox)
    assert res.is_valid is False
    assert any("FACE_TOO_SMALL" in r for r in res.rejection_reasons)


def test_quality_gate_rejects_extreme_yaw():
    """Extreme asymmetric landmarks (large yaw) should be flagged."""
    gate = FaceQualityGate(max_yaw=25.0)
    img = np.ones((150, 150, 3), dtype=np.uint8) * 120
    # Create extreme yaw landmarks where nose is right on top of right eye
    extreme_yaw_lmk = np.array([
        [30.0, 50.0],  # Left eye
        [80.0, 50.0],  # Right eye
        [79.0, 70.0],  # Nose almost at right eye (severe turn)
        [35.0, 90.0],
        [75.0, 90.0],
    ], dtype=np.float32)

    pose = gate.estimate_head_pose_from_landmarks(extreme_yaw_lmk)
    assert abs(pose["yaw"]) > 25.0

    res = gate.assess_quality(img, [10, 10, 140, 140], extreme_yaw_lmk)
    assert res.is_valid is False
    assert any("EXTREME_YAW" in r for r in res.rejection_reasons)
