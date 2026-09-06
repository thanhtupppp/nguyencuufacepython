"""
Tests for 5-point facial landmark alignment using Umeyama similarity transform.
"""

import numpy as np
import pytest
from src.alignment.aligner import (
    FaceAligner,
    ARCFACE_REFERENCE_LANDMARKS_112,
    estimate_similarity_transform_umeyama,
)


def test_umeyama_identity():
    """Testing that aligning canonical landmarks to themselves yields an identity-like transform."""
    template = ARCFACE_REFERENCE_LANDMARKS_112
    m = estimate_similarity_transform_umeyama(template, template)

    # Expected: [[1, 0, 0], [0, 1, 0]]
    expected = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    np.testing.assert_allclose(m, expected, atol=1e-5)


def test_umeyama_translation():
    """Testing pure translation transform."""
    template = ARCFACE_REFERENCE_LANDMARKS_112
    shifted = template + np.array([10.0, -15.0], dtype=np.float32)

    # Mapping shifted -> template
    m = estimate_similarity_transform_umeyama(shifted, template)

    # Transforming shifted points with M should match template
    ones = np.ones((5, 1), dtype=np.float32)
    homogeneous = np.hstack([shifted, ones])  # (5, 3)
    transformed = np.dot(homogeneous, m.T)    # (5, 2)

    np.testing.assert_allclose(transformed, template, atol=1e-4)


def test_face_aligner_output_shape():
    """Testing aligner output image dimensions."""
    aligner = FaceAligner(output_size=(112, 112))
    dummy_img = np.zeros((200, 200, 3), dtype=np.uint8)
    landmarks = ARCFACE_REFERENCE_LANDMARKS_112.copy() + 20.0

    aligned, matrix = aligner.align(dummy_img, landmarks)
    assert aligned.shape == (112, 112, 3)
    assert matrix.shape == (2, 3)
