"""ArcFace-compatible 5-point face alignment.

The canonical template is copied from the official InsightFace Python package
revision pinned in the task documentation. Input landmarks are ordered as
left eye, right eye, nose, left mouth, right mouth.
"""
from __future__ import annotations

import cv2
import numpy as np

ARC_FACE_TEMPLATE_112 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def estimate_similarity_transform(landmarks: np.ndarray) -> np.ndarray:
    """Return a 2x3 similarity transform for five 2D landmarks."""
    points = np.asarray(landmarks, dtype=np.float32)
    if points.shape != (5, 2):
        raise ValueError("landmarks must have shape (5, 2)")
    matrix, _ = cv2.estimateAffinePartial2D(
        points,
        ARC_FACE_TEMPLATE_112,
        method=cv2.LMEDS,
    )
    if matrix is None:
        raise ValueError("unable to estimate similarity transform")
    return matrix.astype(np.float32)


def align_face(image: np.ndarray, landmarks: np.ndarray, output_size: int = 112) -> np.ndarray:
    """Align a BGR image to the ArcFace 112x112 canonical crop."""
    if output_size != 112:
        raise ValueError("this baseline currently supports output_size=112 only")
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be an HxWx3 BGR array")
    matrix = estimate_similarity_transform(landmarks)
    return cv2.warpAffine(
        image,
        matrix,
        (output_size, output_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
