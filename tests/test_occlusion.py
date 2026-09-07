"""
Unit tests for Occlusion and Mask Detection module.
"""

import cv2
import numpy as np
import pytest

from src.quality.occlusion import OcclusionDetector, OcclusionType, OcclusionResult


@pytest.fixture
def detector() -> OcclusionDetector:
    return OcclusionDetector()


def _create_synthetic_face(
    upper_color: tuple[int, int, int] = (120, 150, 210),  # BGR skin tone
    lower_color: tuple[int, int, int] = (120, 150, 210),  # BGR skin tone
    lower_noise_std: float = 5.0,
) -> tuple[np.ndarray, list[int], np.ndarray]:
    """Generates a synthetic 120x120 face image with distinct upper and lower zones."""
    img = np.zeros((120, 120, 3), dtype=np.uint8)
    # Upper half: y in [10, 65]
    img[10:65, 10:110] = upper_color
    # Lower half: y in [65, 110]
    base_lower = np.full((45, 100, 3), lower_color, dtype=np.float32)
    if lower_noise_std > 0:
        noise = np.random.normal(0, lower_noise_std, base_lower.shape)
        base_lower = np.clip(base_lower + noise, 0, 255)
    img[65:110, 10:110] = base_lower.astype(np.uint8)

    bbox = [10, 10, 110, 110]
    # Standard 5 landmarks
    landmarks = np.array([
        [35.0, 45.0],   # left eye
        [85.0, 45.0],   # right eye
        [60.0, 65.0],   # nose tip
        [42.0, 88.0],   # left mouth
        [78.0, 88.0],   # right mouth
    ], dtype=np.float32)

    return img, bbox, landmarks


def test_unoccluded_face_passes(detector: OcclusionDetector) -> None:
    """Natural face with consistent skin tone should pass occlusion checks."""
    img, bbox, lmk = _create_synthetic_face()
    res = detector.detect_occlusion(img, bbox, lmk)

    assert isinstance(res, OcclusionResult)
    assert res.is_occluded is False
    assert res.occlusion_type == OcclusionType.NONE
    assert res.reason is None


def test_blue_surgical_mask_detected(detector: OcclusionDetector) -> None:
    """Lower face covered by cyan/blue surgical mask must be flagged."""
    # BGR for surgical blue/cyan: high Blue (210), medium Green (160), low Red (60)
    img, bbox, lmk = _create_synthetic_face(lower_color=(210, 160, 60))
    res = detector.detect_occlusion(img, bbox, lmk)

    assert res.is_occluded is True
    assert res.occlusion_type == OcclusionType.MASK
    assert res.reason == "SURGICAL_MASK_DETECTED"
    assert res.confidence > 0.6


def test_white_kn95_mask_detected(detector: OcclusionDetector) -> None:
    """Lower face covered by white KN95 mask must be flagged."""
    img, bbox, lmk = _create_synthetic_face(lower_color=(240, 240, 240))
    res = detector.detect_occlusion(img, bbox, lmk)

    assert res.is_occluded is True
    assert res.occlusion_type == OcclusionType.MASK
    assert res.reason == "WHITE_MASK_DETECTED"
    assert res.confidence > 0.6


def test_black_mask_detected(detector: OcclusionDetector) -> None:
    """Lower face covered by smooth dark mask must be flagged."""
    # Low noise std so it's smooth fabric, not a textured beard
    img, bbox, lmk = _create_synthetic_face(lower_color=(25, 25, 25), lower_noise_std=2.0)
    res = detector.detect_occlusion(img, bbox, lmk)

    assert res.is_occluded is True
    assert res.occlusion_type == OcclusionType.MASK
    assert res.reason == "BLACK_MASK_DETECTED"
    assert res.confidence > 0.6


def test_hand_occlusion_landmark_collapse(detector: OcclusionDetector) -> None:
    """Hand covering mouth distorts/collapses mouth landmarks."""
    img, bbox, lmk = _create_synthetic_face()
    # Collapse mouth landmarks: left and right mouth nearly identical
    lmk_collapsed = lmk.copy()
    lmk_collapsed[3] = [60.0, 88.0]
    lmk_collapsed[4] = [62.0, 88.0]  # Width = 2.0 << inter-ocular dist 50.0

    res = detector.detect_occlusion(img, bbox, lmk_collapsed)
    assert res.is_occluded is True
    assert res.occlusion_type == OcclusionType.HAND_OR_OBJECT
    assert "COLLAPSED_MOUTH_LANDMARKS" in (res.reason or "")


def test_hand_occlusion_upper_face(detector: OcclusionDetector) -> None:
    """Hand or dark object covering upper face (eyes/forehead) must be flagged."""
    img, bbox, lmk = _create_synthetic_face(upper_color=(15, 15, 15))
    res = detector.detect_occlusion(img, bbox, lmk)

    assert res.is_occluded is True
    assert res.occlusion_type == OcclusionType.HAND_OR_OBJECT
    assert res.reason == "UPPER_FACE_OCCLUDED"


def test_beard_texture_differentiation(detector: OcclusionDetector) -> None:
    """Dense dark beard has high texture variance and should NOT be falsely tagged as black mask."""
    # High noise std simulates dark facial hair texture
    img, bbox, lmk = _create_synthetic_face(lower_color=(45, 45, 45), lower_noise_std=40.0)
    res = detector.detect_occlusion(img, bbox, lmk)

    # Must not be flagged as black mask because of high Laplacian texture variance
    assert res.reason != "BLACK_MASK_DETECTED"
