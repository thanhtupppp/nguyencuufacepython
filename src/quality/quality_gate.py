"""
Quality Gate module for facial image assessment.
Filters out low-quality faces (blurry, too small, extreme head poses, under/over exposed)
BEFORE passing them to the feature extraction / recognition engine.
"""

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence, Union
import cv2
import numpy as np

from .occlusion import OcclusionDetector, OcclusionResult, OcclusionType


@dataclass
class QualityAssessmentResult:
    """Detailed quality metrics and pass/reject decision."""
    is_valid: bool
    rejection_reasons: list[str] = field(default_factory=list)
    overall_quality_score: float = 1.0  # Normalized [0.0, 1.0]
    blur_score: float = 0.0             # Laplacian variance
    face_size: tuple[int, int] = (0, 0) # (width, height)
    brightness: float = 0.0             # Mean pixel intensity
    estimated_pose: dict[str, float] = field(default_factory=lambda: {"yaw": 0.0, "pitch": 0.0, "roll": 0.0})
    occlusion: Optional[OcclusionResult] = None



class FaceQualityGate:
    """
    Evaluates face image quality based on:
    - Minimum bounding box size
    - Laplacian variance (motion & optical blur)
    - Head pose angles (Yaw, Pitch, Roll) estimated from 5 landmarks
    - Mean illumination & contrast
    """

    def __init__(
        self,
        min_face_size: int = 60,
        optimal_face_size: int = 112,
        blur_threshold: float = 20.0,
        max_yaw: float = 35.0,
        max_pitch: float = 35.0,
        max_roll: float = 30.0,
        min_brightness: float = 30.0,
        max_brightness: float = 230.0,
        check_occlusion: bool = True,
        occlusion_detector: Optional[OcclusionDetector] = None,
    ):
        self.min_face_size = min_face_size
        self.optimal_face_size = optimal_face_size
        self.blur_threshold = blur_threshold
        self.max_yaw = max_yaw
        self.max_pitch = max_pitch
        self.max_roll = max_roll
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.check_occlusion = check_occlusion
        self.occlusion_detector = occlusion_detector or (OcclusionDetector() if check_occlusion else None)

    def estimate_head_pose_from_landmarks(self, landmarks: np.ndarray) -> dict[str, float]:
        """
        Approximates Yaw, Pitch, Roll angles (in degrees) using 5 canonical landmarks:
        0: left eye, 1: right eye, 2: nose tip, 3: left mouth, 4: right mouth.
        """
        lm = np.asarray(landmarks, dtype=np.float32)
        assert lm.shape == (5, 2), "Landmarks must be (5, 2)"

        left_eye, right_eye, nose, left_mouth, right_mouth = lm[0], lm[1], lm[2], lm[3], lm[4]

        # 1. Roll: In-plane rotation angle of the eye line
        d_eye_x = right_eye[0] - left_eye[0]
        d_eye_y = right_eye[1] - left_eye[1]
        roll = float(np.degrees(np.arctan2(d_eye_y, d_eye_x)))

        # 2. Yaw: Left vs Right eye-to-nose horizontal distance symmetry
        dist_left_eye_nose = np.linalg.norm(nose - left_eye)
        dist_right_eye_nose = np.linalg.norm(nose - right_eye)
        total_eye_dist = dist_left_eye_nose + dist_right_eye_nose

        if total_eye_dist > 1e-6:
            sym_ratio = (dist_right_eye_nose - dist_left_eye_nose) / total_eye_dist
            yaw = float(np.clip(sym_ratio * 75.0, -90.0, 90.0))
        else:
            yaw = 0.0

        # 3. Pitch: Vertical ratio between eye-nose distance and eye-mouth distance
        eye_center_y = (left_eye[1] + right_eye[1]) / 2.0
        mouth_center_y = (left_mouth[1] + right_mouth[1]) / 2.0
        total_facial_height = mouth_center_y - eye_center_y
        v_eye_nose = nose[1] - eye_center_y

        if total_facial_height > 1e-6:
            # In canonical frontal faces, nose tip is ~55% between eye-line and mouth-line
            vert_ratio = v_eye_nose / total_facial_height
            pitch = float(np.clip((vert_ratio - 0.55) * 60.0, -90.0, 90.0))
        else:
            pitch = 0.0

        return {"yaw": yaw, "pitch": pitch, "roll": roll}

    def assess_blur(self, face_crop_gray: np.ndarray) -> float:
        """Computes variance of Laplacian on grayscale face crop."""
        laplacian = cv2.Laplacian(face_crop_gray, cv2.CV_64F)
        return float(laplacian.var())

    def assess_quality(
        self,
        image: np.ndarray,
        bbox: Union[Sequence[Union[float, int]], np.ndarray],
        landmarks: Optional[np.ndarray] = None,
    ) -> QualityAssessmentResult:
        """
        Assesses face quality against established thresholds.

        :param image: Original BGR image
        :param bbox: [x1, y1, x2, y2]
        :param landmarks: (5, 2) facial keypoints (optional)
        :return: QualityAssessmentResult
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h_img, w_img = image.shape[:2]

        # Clip crop coordinates
        x1_c = max(0, x1)
        y1_c = max(0, y1)
        x2_c = min(w_img, x2)
        y2_c = min(h_img, y2)

        face_w = x2_c - x1_c
        face_h = y2_c - y1_c

        rejection_reasons = []

        # 1. Size Check
        if face_w < self.min_face_size or face_h < self.min_face_size:
            rejection_reasons.append(f"FACE_TOO_SMALL (got {face_w}x{face_h}, min {self.min_face_size})")

        if face_w <= 0 or face_h <= 0:
            return QualityAssessmentResult(
                is_valid=False,
                rejection_reasons=["INVALID_BBOX_COORDINATES"],
                overall_quality_score=0.0,
            )

        face_crop = image[y1_c:y2_c, x1_c:x2_c]
        face_gray: np.ndarray = np.asarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY) if face_crop.ndim == 3 else face_crop)

        # 2. Blur Check
        blur_score = self.assess_blur(face_gray)
        if blur_score < self.blur_threshold:
            rejection_reasons.append(f"BLURRY_FACE (var {blur_score:.1f} < threshold {self.blur_threshold:.1f})")

        # 3. Illumination / Brightness Check
        mean_brightness = float(np.mean(face_gray))
        if mean_brightness < self.min_brightness:
            rejection_reasons.append(f"UNDER_EXPOSED (brightness {mean_brightness:.1f} < min {self.min_brightness:.1f})")
        elif mean_brightness > self.max_brightness:
            rejection_reasons.append(f"OVER_EXPOSED (brightness {mean_brightness:.1f} > max {self.max_brightness:.1f})")

        # 4. Pose Check (if landmarks provided)
        pose = {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}
        if landmarks is not None:
            pose = self.estimate_head_pose_from_landmarks(landmarks)
            if abs(pose["yaw"]) > self.max_yaw:
                rejection_reasons.append(f"EXTREME_YAW (|yaw| {abs(pose['yaw']):.1f}° > {self.max_yaw}°)")
            if abs(pose["pitch"]) > self.max_pitch:
                rejection_reasons.append(f"EXTREME_PITCH (|pitch| {abs(pose['pitch']):.1f}° > {self.max_pitch}°)")
            if abs(pose["roll"]) > self.max_roll:
                rejection_reasons.append(f"EXTREME_ROLL (|roll| {abs(pose['roll']):.1f}° > {self.max_roll}°)")

        # 5. Occlusion & Mask Check
        occlusion_res: Optional[OcclusionResult] = None
        if self.check_occlusion and self.occlusion_detector is not None:
            occlusion_res = self.occlusion_detector.detect_occlusion(image, bbox, landmarks)
            if occlusion_res.is_occluded:
                if occlusion_res.occlusion_type == OcclusionType.MASK:
                    rejection_reasons.append(f"MASK_DETECTED ({occlusion_res.reason})")
                else:
                    rejection_reasons.append(f"OCCLUDED_FACE ({occlusion_res.reason})")

        # Compute overall quality score [0.0, 1.0]
        size_score = min(1.0, min(face_w, face_h) / float(self.optimal_face_size))
        blur_factor = min(1.0, blur_score / max(self.blur_threshold * 2.0, 1.0))
        pose_penalty = 1.0 - min(1.0, (abs(pose["yaw"]) + abs(pose["pitch"])) / 90.0)
        overall_score = float(np.clip(0.4 * blur_factor + 0.3 * size_score + 0.3 * pose_penalty, 0.0, 1.0))
        if occlusion_res is not None and occlusion_res.is_occluded:
            overall_score = 0.0

        is_valid = len(rejection_reasons) == 0

        return QualityAssessmentResult(
            is_valid=is_valid,
            rejection_reasons=rejection_reasons,
            overall_quality_score=overall_score,
            blur_score=blur_score,
            face_size=(face_w, face_h),
            brightness=mean_brightness,
            estimated_pose=pose,
            occlusion=occlusion_res,
        )
