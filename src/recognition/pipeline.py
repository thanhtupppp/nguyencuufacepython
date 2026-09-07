"""End-to-end single-image recognition pipeline.

The pipeline is intentionally model-asset agnostic: callers provide configured
SCRFD and ArcFace ONNX components. It enforces the production order
SCRFD -> quality gate -> 5-point alignment -> ArcFace embedding and returns
metadata needed by enrollment/recognition APIs.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.alignment.aligner import FaceAligner
from src.detection.scrfd import SCRFDDetector
from src.quality.quality_gate import FaceQualityGate
from .base import BaseFaceRecognizer


@dataclass(frozen=True)
class FaceEmbeddingResult:
    embedding: np.ndarray
    model_version: str
    quality_score: float
    bbox: list[float]
    landmarks: np.ndarray
    detector_score: float


class RecognitionPipeline:
    def __init__(
        self,
        detector: SCRFDDetector,
        recognizer: BaseFaceRecognizer,
        aligner: Optional[FaceAligner] = None,
        quality_gate: Optional[FaceQualityGate] = None,
    ):
        self.detector = detector
        self.recognizer = recognizer
        self.aligner = aligner or FaceAligner()
        self.quality_gate = quality_gate or FaceQualityGate()

    def extract_best_face(self, image: np.ndarray) -> FaceEmbeddingResult:
        detections = self.detector.detect(image)
        if not detections:
            raise ValueError("NO_FACE_DETECTED")

        # For single-person enrollment/recognition, choose the highest detector
        # confidence among valid quality candidates. Multi-face workflows should
        # call detect() and process each detection explicitly.
        candidates = []
        last_rejections = []
        for detection in detections:
            landmarks = detection.get("landmarks")
            if landmarks is None:
                continue
            quality = self.quality_gate.assess_quality(
                image, detection["bbox"], np.asarray(landmarks, dtype=np.float32)
            )
            if quality.is_valid:
                candidates.append((float(detection["score"]), detection, quality))
            else:
                last_rejections.extend(getattr(quality, "rejection_reasons", []))

        if not candidates:
            if any("MASK_DETECTED" in r for r in last_rejections):
                raise ValueError("MASK_DETECTED")
            if any("OCCLUDED_FACE" in r for r in last_rejections):
                raise ValueError("OCCLUSION_DETECTED")
            raise ValueError("NO_FACE_PASSED_QUALITY_GATE")

        _, detection, quality = max(candidates, key=lambda item: item[0])
        aligned, _ = self.aligner.align(
            image, np.asarray(detection["landmarks"], dtype=np.float32)
        )
        embedding = self.recognizer.extract_embedding(aligned)

        if embedding.shape != (self.recognizer.embedding_dim,):
            raise ValueError(
                f"INVALID_EMBEDDING_SHAPE: expected {(self.recognizer.embedding_dim,)}, got {embedding.shape}"
            )
        if not np.isfinite(embedding).all():
            raise ValueError("INVALID_EMBEDDING_VALUES")

        return FaceEmbeddingResult(
            embedding=embedding,
            model_version=self.recognizer.model_version,
            quality_score=quality.overall_quality_score,
            bbox=list(detection["bbox"]),
            landmarks=np.asarray(detection["landmarks"], dtype=np.float32),
            detector_score=float(detection["score"]),
        )
