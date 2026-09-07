"""
Face alignment module using 5 facial landmarks and Umeyama similarity transformation.
Standard output size: 112 x 112 pixels (canonical ArcFace / InsightFace format).
"""

from typing import Optional
import cv2
import numpy as np


# Standard 5-point reference landmarks for 112x112 aligned face
# Format: [left_eye, right_eye, nose_tip, left_mouth_corner, right_mouth_corner]
ARCFACE_REFERENCE_LANDMARKS_112 = np.array(
    [
        [38.2946, 51.6963],  # Left eye
        [73.5318, 51.5014],  # Right eye
        [56.0252, 71.7366],  # Nose tip
        [41.5493, 92.3655],  # Left mouth corner
        [70.7299, 92.2041],  # Right mouth corner
    ],
    dtype=np.float32,
)


def estimate_similarity_transform_umeyama(
    src: np.ndarray, dst: np.ndarray
) -> np.ndarray:
    """
    Computes optimal 2D similarity transform matrix M (2x3) mapping src to dst.
    s * R * src + t = dst
    Based on Shinji Umeyama (1991): "Least-squares estimation of transformation parameters
    between two point patterns", IEEE TPAMI.

    :param src: (N, 2) source landmarks
    :param dst: (N, 2) destination landmarks (reference template)
    :return: 2x3 affine transformation matrix
    """
    assert src.shape == dst.shape and src.shape[1] == 2, "Points must be (N, 2)"
    num_points = src.shape[0]

    # Compute centroids
    mean_src = np.mean(src, axis=0)
    mean_dst = np.mean(dst, axis=0)

    # Center the points
    src_centered = src - mean_src
    dst_centered = dst - mean_dst

    # Variance of source points
    var_src = np.mean(np.sum(src_centered**2, axis=1))
    if var_src < 1e-8:
        # Avoid division by zero
        return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

    # Covariance matrix between dst and src
    covariance = np.dot(dst_centered.T, src_centered) / num_points

    # Singular Value Decomposition
    u, d, vt = np.linalg.svd(covariance)

    # Reflection check
    det_cov = np.linalg.det(covariance)
    s = np.eye(2, dtype=np.float32)
    if det_cov < 0 or (det_cov == 0 and np.linalg.det(u) * np.linalg.det(vt) < 0):
        s[1, 1] = -1.0

    # Rotation matrix R
    rotation = np.dot(u, np.dot(s, vt))

    # Scale factor c
    scale = np.sum(d * np.diag(s)) / var_src

    # Translation vector t
    translation = mean_dst - scale * np.dot(rotation, mean_src)

    # 2x3 transformation matrix
    transform_matrix = np.zeros((2, 3), dtype=np.float32)
    transform_matrix[:2, :2] = scale * rotation
    transform_matrix[:2, 2] = translation

    return transform_matrix


class FaceAligner:
    """
    Aligns and crops facial images to 112x112 canonical representation
    using 5 facial landmarks.
    """

    def __init__(
        self,
        output_size: tuple[int, int] = (112, 112),
        reference_landmarks: Optional[np.ndarray] = None,
    ):
        self.output_size = output_size
        if reference_landmarks is None:
            if output_size == (112, 112):
                self.reference_landmarks = ARCFACE_REFERENCE_LANDMARKS_112.copy()
            else:
                # Scale landmarks if output size differs from 112x112
                scale_x = output_size[0] / 112.0
                scale_y = output_size[1] / 112.0
                scaled = ARCFACE_REFERENCE_LANDMARKS_112.copy()
                scaled[:, 0] *= scale_x
                scaled[:, 1] *= scale_y
                self.reference_landmarks = scaled
        else:
            self.reference_landmarks = reference_landmarks.astype(np.float32)

    def align(
        self, image: np.ndarray, landmarks: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Align face image to standard canonical template.

        :param image: Input BGR image (H, W, C)
        :param landmarks: 5 landmarks array of shape (5, 2)
        :return: (aligned_face_112x112, 2x3 transformation matrix)
        """
        landmarks = np.asarray(landmarks, dtype=np.float32)
        assert landmarks.shape == (5, 2), f"Expected (5, 2) landmarks, got {landmarks.shape}"

        m = estimate_similarity_transform_umeyama(landmarks, self.reference_landmarks)
        aligned = cv2.warpAffine(
            image,
            m,
            self.output_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0.0, 0.0, 0.0),
        )
        return aligned, m
