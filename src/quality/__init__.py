"""
Quality assessment module: Blur, size, pose (yaw/pitch/roll), illumination, and occlusion checks.
"""

from .occlusion import OcclusionDetector, OcclusionResult, OcclusionType
from .quality_gate import FaceQualityGate, QualityAssessmentResult

__all__ = [
    "FaceQualityGate",
    "QualityAssessmentResult",
    "OcclusionDetector",
    "OcclusionResult",
    "OcclusionType",
]

