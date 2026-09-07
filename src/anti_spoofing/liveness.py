"""
Anti-Spoofing (Liveness Detection) Module.
Prevents presentation attacks (printed photos, screen replay attacks, 2D paper masks)
using multi-scale crop analysis and deep learning liveness models (MiniFASNet).
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union
import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None


@dataclass
class LivenessResult:
    """Outcome of liveness verification."""
    is_live: bool
    liveness_score: float                  # Probability of live human [0.0, 1.0]
    threshold: float                       # Operating decision threshold
    attack_type: Optional[str] = None      # 'PRINT_ATTACK', 'SCREEN_REPLAY', or None
    scale_scores: dict[str, float] = field(default_factory=dict)
    fourier_score: float = 1.0             # High-frequency texture consistency score


def crop_face_with_scale(
    image: np.ndarray,
    bbox: list[float] | np.ndarray,
    scale: float,
    output_size: tuple[int, int] = (80, 80),
) -> np.ndarray:
    """
    Crops face region expanded by a scale factor with boundary padding.
    Scale ~1.0: Tight facial crop.
    Scale ~2.7: Wide context crop (capturing screen bezel, paper edges, holding fingers).
    """
    x1, y1, x2, y2 = [float(v) for v in bbox]
    w = x2 - x1
    h = y2 - y1

    center_x = x1 + w / 2.0
    center_y = y1 + h / 2.0

    # Expand box by scale
    scaled_w = w * scale
    scaled_h = h * scale

    new_x1 = round(center_x - scaled_w / 2.0)
    new_y1 = round(center_y - scaled_h / 2.0)
    new_x2 = round(center_x + scaled_w / 2.0)
    new_y2 = round(center_y + scaled_h / 2.0)

    img_h, img_w = image.shape[:2]

    # Calculate padding if crop goes outside image boundaries
    pad_left = max(0, -new_x1)
    pad_top = max(0, -new_y1)
    pad_right = max(0, new_x2 - img_w)
    pad_bottom = max(0, new_y2 - img_h)

    crop_x1 = max(0, new_x1)
    crop_y1 = max(0, new_y1)
    crop_x2 = min(img_w, new_x2)
    crop_y2 = min(img_h, new_y2)

    crop = image[crop_y1:crop_y2, crop_x1:crop_x2]

    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        crop = cv2.copyMakeBorder(
            crop, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )

    if crop.shape[:2] != output_size:
        crop = cv2.resize(crop, output_size, interpolation=cv2.INTER_LINEAR)

    return crop


def compute_fourier_frequency_score(face_gray: np.ndarray) -> float:
    """
    Analyzes 2D Fourier frequency spectrum.
    Electronic screens and paper prints often have abnormal high-frequency peaks
    (Moiré patterns / print dot matrices) or severe high-frequency loss.
    Returns texture naturalness score [0.0, 1.0].
    """
    if face_gray.size == 0:
        return 0.0

    h, w = face_gray.shape[:2]
    # Resize to standard analysis size
    resized = cv2.resize(face_gray, (64, 64)) if (h, w) != (64, 64) else face_gray

    # Compute 2D Fast Fourier Transform
    f = np.fft.fft2(resized.astype(np.float32))
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.log(np.abs(fshift) + 1.0)

    # Separate low frequencies (center) from high frequencies (outer)
    cy, cx = 32, 32
    r_inner = 8
    y, x = np.ogrid[:64, :64]
    mask_inner = ((x - cx) ** 2 + (y - cy) ** 2) <= r_inner**2

    low_freq_energy = np.mean(magnitude_spectrum[mask_inner])
    high_freq_energy = np.mean(magnitude_spectrum[~mask_inner])

    if low_freq_energy <= 1e-6:
        return 0.5

    ratio = high_freq_energy / low_freq_energy
    # Natural human skin typically exhibits a balanced ratio between 0.35 and 0.75
    if 0.30 <= ratio <= 0.80:
        return 1.0
    elif 0.20 <= ratio < 0.30 or 0.80 < ratio <= 0.90:
        return 0.6
    else:
        return 0.2


class AntiSpoofDetector:
    """
    MiniFASNet Liveness Detector running on ONNX Runtime.
    Evaluates faces on multiple scales (e.g. 1.0x and 2.7x) to capture both
    micro-skin reflections and macro-environment context (screen borders, paper edges).
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        threshold: float = 0.85,
        scales: list[float] = [1.0, 2.7],
        providers: Optional[list[str]] = None,
    ):
        self.model_path = Path(model_path) if model_path else None
        self.threshold = threshold
        self.scales = scales
        self.session: Any = None

        if providers is None:
            available = ort.get_available_providers() if ort else []
            self.providers = [
                p
                for p in ["CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]
                if p in available
            ]
            if not self.providers and ort:
                self.providers = ["CPUExecutionProvider"]
        else:
            self.providers = providers

        if self.model_path and self.model_path.exists():
            self._load_model()

    def _load_model(self) -> None:
        if ort is None:
            raise ImportError("onnxruntime is required for AntiSpoofDetector")

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Cap CPU threads to prevent thrashing high-core CPUs (e.g. 48 cores)
        session_options.intra_op_num_threads = min(4, os.cpu_count() or 4)
        session_options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(self.model_path), sess_options=session_options, providers=self.providers
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape  # typically [1, 3, 80, 80]

    def _preprocess_crop(self, crop: np.ndarray) -> np.ndarray:
        """Prepares crop for MiniFASNet input (NCHW float32)."""
        # Normalization: typically (crop - mean) / std or crop / 255.0
        blob = crop.astype(np.float32)
        blob = np.transpose(blob, (2, 0, 1))  # (3, 80, 80)
        blob = np.expand_dims(blob, axis=0)    # (1, 3, 80, 80)
        return blob

    def predict_liveness(
        self,
        image: np.ndarray,
        bbox: list[float] | np.ndarray,
    ) -> LivenessResult:
        """
        Determines whether the face in bbox is a real live person or a spoof attack.
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h_img, w_img = image.shape[:2]
        crop_tight = image[max(0, y1):min(h_img, y2), max(0, x1):min(w_img, x2)]

        # 1. Fourier texture analysis
        gray_tight = (
            cv2.cvtColor(crop_tight, cv2.COLOR_BGR2GRAY)
            if crop_tight.ndim == 3 and crop_tight.size > 0
            else crop_tight
        )
        fourier_score = compute_fourier_frequency_score(gray_tight)

        # 2. Deep learning multi-scale inference
        scale_scores = {}
        if self.session is not None:
            for scale in self.scales:
                scaled_crop = crop_face_with_scale(image, bbox, scale=scale, output_size=(80, 80))
                tensor = self._preprocess_crop(scaled_crop)
                raw_out = self.session.run([self.output_name], {self.input_name: tensor})[0]
                # Apply softmax over class logits
                exp_scores = np.exp(raw_out - np.max(raw_out, axis=1, keepdims=True))
                probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
                # Class 1 is live probability
                live_prob = float(probs[0, 1]) if probs.shape[1] > 1 else float(probs[0, 0])
                scale_scores[f"scale_{scale:.1f}"] = live_prob

            overall_liveness = float(np.mean(list(scale_scores.values())))
        else:
            # Heuristic fallback if deep learning weights are not yet mounted
            overall_liveness = fourier_score
            scale_scores = {"fourier_heuristic": fourier_score}

        is_live = overall_liveness >= self.threshold

        attack_type = None
        if not is_live:
            # Classify attack type based on frequency & scale disparity
            if fourier_score < 0.5:
                attack_type = "SCREEN_REPLAY_MOIRE"
            else:
                attack_type = "PRINT_ATTACK"

        return LivenessResult(
            is_live=is_live,
            liveness_score=overall_liveness,
            threshold=self.threshold,
            attack_type=attack_type,
            scale_scores=scale_scores,
            fourier_score=fourier_score,
        )
