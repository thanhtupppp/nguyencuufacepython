import numpy as np
import pytest

from src.alignment.arcface import ARC_FACE_TEMPLATE_112, estimate_similarity_transform


def test_template_is_canonical_112x112():
    assert ARC_FACE_TEMPLATE_112.shape == (5, 2)
    assert np.allclose(ARC_FACE_TEMPLATE_112[0], [38.2946, 51.6963])
    assert np.allclose(ARC_FACE_TEMPLATE_112[4], [70.7299, 92.2041])


def test_identity_landmarks_produce_identity_transform():
    matrix = estimate_similarity_transform(ARC_FACE_TEMPLATE_112)
    assert matrix.shape == (2, 3)
    assert np.allclose(matrix[:, :2], np.eye(2), atol=1e-4)
    assert np.allclose(matrix[:, 2], [0, 0], atol=1e-4)


def test_invalid_landmark_shape_fails_closed():
    with pytest.raises(ValueError):
        estimate_similarity_transform(np.zeros((4, 2), dtype=np.float32))
