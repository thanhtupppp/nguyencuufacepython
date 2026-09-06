"""
Anti-Spoofing and Liveness Detection subpackage.
"""

from .liveness import (
    AntiSpoofDetector,
    LivenessResult,
    crop_face_with_scale,
    compute_fourier_frequency_score,
)

__all__ = [
    "AntiSpoofDetector",
    "LivenessResult",
    "crop_face_with_scale",
    "compute_fourier_frequency_score",
]
