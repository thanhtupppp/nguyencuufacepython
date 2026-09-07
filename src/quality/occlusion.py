"""
Occlusion and Mask Detection module for Face Quality Gate.
Detects face masks (surgical, KN95, cloth, black/white/blue) and hand/object occlusion.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Sequence, Union
import cv2
import numpy as np


class OcclusionType(str, Enum):
    NONE = "NONE"
    MASK = "MASK"
    HAND_OR_OBJECT = "HAND_OR_OBJECT"


@dataclass
class OcclusionResult:
    """Detailed occlusion and mask detection assessment."""
    is_occluded: bool
    occlusion_type: OcclusionType = OcclusionType.NONE
    confidence: float = 0.0
    reason: Optional[str] = None
    details: dict[str, float] = field(default_factory=dict)


class OcclusionDetector:
    """
    Multi-cue Occlusion & Mask Detector:
    1. Upper vs Lower Face Skin Discordance (YCrCb + HSV).
    2. Surgical / KN95 / Dark Mask Color Signature Detection.
    3. Beard vs Mask discrimination via texture variance.
    4. Landmark Geometric Proportions & Hand Occlusion Distortion.
    """

    def __init__(
        self,
        min_upper_skin_ratio: float = 0.30,
        skin_discordance_threshold: float = 0.32,
        mask_blue_threshold: float = 0.15,
        mask_white_threshold: float = 0.40,
        mask_dark_threshold: float = 0.45,
        mouth_aspect_min: float = 0.35,
        mouth_aspect_max: float = 1.25,
    ):
        self.min_upper_skin_ratio = min_upper_skin_ratio
        self.skin_discordance_threshold = skin_discordance_threshold
        self.mask_blue_threshold = mask_blue_threshold
        self.mask_white_threshold = mask_white_threshold
        self.mask_dark_threshold = mask_dark_threshold
        self.mouth_aspect_min = mouth_aspect_min
        self.mouth_aspect_max = mouth_aspect_max

    @staticmethod
    def _extract_skin_mask(bgr_img: np.ndarray) -> np.ndarray:
        """
        Creates a binary mask where 255 indicates human skin tone.
        Combines YCrCb bounds with HSV bounds for high robustness across diverse skin tones.
        """
        ycrcb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YCrCb)
        hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)

        # YCrCb skin cluster
        lower_ycrcb = np.array([20, 133, 77], dtype=np.uint8)
        upper_ycrcb = np.array([255, 175, 127], dtype=np.uint8)
        mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)

        # HSV skin cluster (hue ~ 0-28 or 168-180, saturation >= 20, value >= 35)
        lower_hsv1 = np.array([0, 20, 35], dtype=np.uint8)
        upper_hsv1 = np.array([28, 255, 255], dtype=np.uint8)
        mask_hsv1 = cv2.inRange(hsv, lower_hsv1, upper_hsv1)

        lower_hsv2 = np.array([168, 20, 35], dtype=np.uint8)
        upper_hsv2 = np.array([180, 255, 255], dtype=np.uint8)
        mask_hsv2 = cv2.inRange(hsv, lower_hsv2, upper_hsv2)

        mask_hsv = cv2.bitwise_or(mask_hsv1, mask_hsv2)
        return cv2.bitwise_and(mask_ycrcb, mask_hsv)

    def detect_occlusion(
        self,
        image: np.ndarray,
        bbox: Union[Sequence[Union[float, int]], np.ndarray],
        landmarks: Optional[np.ndarray] = None,
    ) -> OcclusionResult:
        """
        Evaluates whether the face is occluded or wearing a mask.
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h_img, w_img = image.shape[:2]

        x1_c = max(0, x1)
        y1_c = max(0, y1)
        x2_c = min(w_img, x2)
        y2_c = min(h_img, y2)

        face_w = x2_c - x1_c
        face_h = y2_c - y1_c

        if face_w < 20 or face_h < 20:
            return OcclusionResult(
                is_occluded=False,
                occlusion_type=OcclusionType.NONE,
                confidence=0.0,
                reason=None,
            )

        face_bgr = image[y1_c:y2_c, x1_c:x2_c]

        # 1. Partition face into Upper ROI (eyes/forehead) and Lower ROI (nose/mouth/chin)
        if landmarks is not None and len(landmarks) == 5:
            # Map landmarks to crop local coordinates
            lm_local = landmarks - np.array([x1_c, y1_c], dtype=np.float32)
            eye_y = float(np.mean([lm_local[0][1], lm_local[1][1]]))
            nose_y = float(lm_local[2][1])
            split_y = int(np.clip(nose_y, 0.35 * face_h, 0.70 * face_h))
        else:
            split_y = int(0.50 * face_h)

        upper_bgr = face_bgr[:split_y, :]
        lower_bgr = face_bgr[split_y:, :]

        if upper_bgr.size == 0 or lower_bgr.size == 0:
            return OcclusionResult(is_occluded=False, occlusion_type=OcclusionType.NONE)

        # 2. Skin ratio analysis
        upper_skin_mask = self._extract_skin_mask(upper_bgr)
        lower_skin_mask = self._extract_skin_mask(lower_bgr)

        upper_skin_ratio = float(np.mean(upper_skin_mask > 0))
        lower_skin_ratio = float(np.mean(lower_skin_mask > 0))
        skin_discordance = upper_skin_ratio - lower_skin_ratio

        # 3. Check for Blue/Cyan Surgical Mask
        hsv_lower = cv2.cvtColor(lower_bgr, cv2.COLOR_BGR2HSV)
        h_chan, s_chan, v_chan = cv2.split(hsv_lower)
        # Blue/Cyan mask: Hue in [80, 135], S >= 35, V >= 40
        blue_mask = (h_chan >= 80) & (h_chan <= 135) & (s_chan >= 35) & (v_chan >= 40)
        blue_ratio = float(np.mean(blue_mask))

        # 4. Check for White / KN95 Mask
        # Low saturation, high brightness
        white_mask = (s_chan < 32) & (v_chan > 155)
        white_ratio = float(np.mean(white_mask))

        # 5. Check for Black / Dark Mask
        dark_mask = v_chan < 45
        dark_ratio = float(np.mean(dark_mask))

        # 6. Lower face texture analysis (Laplacian variance on inner core) to differentiate beard vs smooth mask
        lower_gray = cv2.cvtColor(lower_bgr, cv2.COLOR_BGR2GRAY)
        margin_y = max(1, int(0.12 * lower_gray.shape[0]))
        margin_x = max(1, int(0.12 * lower_gray.shape[1]))
        lower_core = lower_gray[margin_y:-margin_y, margin_x:-margin_x]
        if lower_core.size > 0:
            lower_texture_var = float(cv2.Laplacian(lower_core, cv2.CV_64F).var())
        else:
            lower_texture_var = float(cv2.Laplacian(lower_gray, cv2.CV_64F).var())

        # Check if texture indicates hair/beard (beards have lots of fine edges, masks are smooth)
        is_beard_likely = lower_texture_var > 60.0 and blue_ratio < 0.08 and white_ratio < 0.15

        # 7. Check Landmark Geometry for Hand Occlusion
        landmark_anomaly = False
        landmark_reason = None
        if landmarks is not None and len(landmarks) == 5:
            left_eye, right_eye, nose, left_mouth, right_mouth = landmarks
            inter_ocular = float(np.linalg.norm(right_eye - left_eye))
            mouth_width = float(np.linalg.norm(right_mouth - left_mouth))
            nose_to_mouth = float(np.linalg.norm((left_mouth + right_mouth) / 2.0 - nose))

            if inter_ocular > 1e-4:
                mouth_ratio = mouth_width / inter_ocular
                nose_mouth_ratio = nose_to_mouth / inter_ocular

                # Abnormal distortion often caused by hand or partial occlusion
                if mouth_ratio < self.mouth_aspect_min:
                    landmark_anomaly = True
                    landmark_reason = f"COLLAPSED_MOUTH_LANDMARKS (ratio {mouth_ratio:.2f} < {self.mouth_aspect_min:.2f})"
                elif mouth_ratio > self.mouth_aspect_max:
                    landmark_anomaly = True
                    landmark_reason = f"DISTORTED_MOUTH_LANDMARKS (ratio {mouth_ratio:.2f} > {self.mouth_aspect_max:.2f})"
                elif nose_mouth_ratio < 0.20:
                    landmark_anomaly = True
                    landmark_reason = f"NOSE_MOUTH_OVERLAP (ratio {nose_mouth_ratio:.2f} < 0.20)"

        details = {
            "upper_skin_ratio": round(upper_skin_ratio, 3),
            "lower_skin_ratio": round(lower_skin_ratio, 3),
            "skin_discordance": round(skin_discordance, 3),
            "blue_mask_ratio": round(blue_ratio, 3),
            "white_mask_ratio": round(white_ratio, 3),
            "dark_mask_ratio": round(dark_ratio, 3),
            "lower_texture_var": round(lower_texture_var, 1),
        }

        # --- Evaluate Decision Rules ---

        # Rule 1: Blue / Cyan surgical mask
        # Requires genuine skin in upper face AND low skin in lower face OR dominant blue ratio
        if (
            (upper_skin_ratio >= self.min_upper_skin_ratio and lower_skin_ratio < 0.30 and blue_ratio >= self.mask_blue_threshold)
            or (blue_ratio >= 0.40)
        ):
            return OcclusionResult(
                is_occluded=True,
                occlusion_type=OcclusionType.MASK,
                confidence=min(1.0, blue_ratio * 3.0),
                reason="SURGICAL_MASK_DETECTED",
                details=details,
            )

        # Rule 2: White / KN95 mask (high white ratio with low skin ratio in lower face)
        if white_ratio >= self.mask_white_threshold and lower_skin_ratio < 0.30:
            return OcclusionResult(
                is_occluded=True,
                occlusion_type=OcclusionType.MASK,
                confidence=min(1.0, white_ratio * 1.8),
                reason="WHITE_MASK_DETECTED",
                details=details,
            )

        # Rule 3: Black / Dark mask (smooth dark region without beard texture)
        if dark_ratio >= self.mask_dark_threshold and not is_beard_likely:
            return OcclusionResult(
                is_occluded=True,
                occlusion_type=OcclusionType.MASK,
                confidence=min(1.0, dark_ratio * 1.5),
                reason="BLACK_MASK_DETECTED",
                details=details,
            )

        # Rule 4: General Mask / Cloth via Skin Discordance
        # Upper face has normal skin, but lower face has abnormally low skin, and it's not a beard
        if (
            upper_skin_ratio >= self.min_upper_skin_ratio
            and skin_discordance >= self.skin_discordance_threshold
            and lower_skin_ratio < 0.22
            and not is_beard_likely
        ):
            return OcclusionResult(
                is_occluded=True,
                occlusion_type=OcclusionType.MASK,
                confidence=min(1.0, skin_discordance * 1.6),
                reason="MASK_DETECTED",
                details=details,
            )

        # Rule 5: Hand covering face (Landmark Geometric Anomaly)
        if landmark_anomaly:
            return OcclusionResult(
                is_occluded=True,
                occlusion_type=OcclusionType.HAND_OR_OBJECT,
                confidence=0.85,
                reason=landmark_reason,
                details=details,
            )

        # Rule 6: Hand covering face (Upper face occluded while lower face detected)
        if upper_skin_ratio < 0.12 and lower_skin_ratio > 0.35:
            return OcclusionResult(
                is_occluded=True,
                occlusion_type=OcclusionType.HAND_OR_OBJECT,
                confidence=0.80,
                reason="UPPER_FACE_OCCLUDED",
                details=details,
            )

        return OcclusionResult(
            is_occluded=False,
            occlusion_type=OcclusionType.NONE,
            confidence=0.0,
            reason=None,
            details=details,
        )
