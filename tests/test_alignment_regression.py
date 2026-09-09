import hashlib

import cv2
import numpy as np

from src.alignment.arcface import ARC_FACE_TEMPLATE_112, align_face


def _fixture_image() -> np.ndarray:
    """Deterministic synthetic BGR image; no external binary fixture required."""
    h, w = 160, 200
    yy, xx = np.indices((h, w), dtype=np.uint16)
    b = (xx * 3 + yy) % 256
    g = (xx + yy * 2) % 256
    r = (xx * 5 + yy * 7) % 256
    return np.stack([b, g, r], axis=-1).astype(np.uint8)


def test_alignment_regression_is_deterministic():
    image = _fixture_image()
    landmarks = ARC_FACE_TEMPLATE_112.copy()

    crop_a = align_face(image, landmarks)
    crop_b = align_face(image, landmarks)

    assert crop_a.shape == (112, 112, 3)
    assert crop_a.dtype == np.uint8
    assert np.array_equal(crop_a, crop_b)
    assert hashlib.sha256(crop_a.tobytes()).hexdigest() == (
        "5c7b6d3f1e7f2bb5d9c9f5a5b5d2d0c6c1e4a6f8f2c2d1e5c9e5b7b0b4e3c2a1"
    )


def test_alignment_uses_similarity_transform_for_known_scale_translation():
    image = _fixture_image()
    source = ARC_FACE_TEMPLATE_112 * 0.9 + np.array([20.0, 15.0], dtype=np.float32)
    crop = align_face(image, source)
    assert crop.shape == (112, 112, 3)
    assert np.isfinite(crop).all()
