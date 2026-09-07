"""Build the fail-closed recognition runtime for API processes."""

import os
from pathlib import Path

from src.alignment.aligner import FaceAligner
from src.detection.scrfd import SCRFDDetector
from src.quality.quality_gate import FaceQualityGate
from src.recognition.arcface import ArcFaceRecognizer
from src.recognition.model_registry import ModelRegistry, ModelSpec
from src.recognition.pipeline import RecognitionPipeline


def build_recognition_pipeline() -> RecognitionPipeline:
    """Create the production pipeline from environment configuration.

    No model asset or fingerprint means no API recognition service is created.
    This keeps enrollment/recognition fail-closed until real licensed assets are
    installed and explicitly fingerprinted.
    """
    model_version = os.getenv("FACE_MODEL_VERSION", "arcface_v1")
    arcface_path = Path(os.getenv("ARCFACE_MODEL_PATH", "models/arcface.onnx"))
    expected_sha = os.getenv("ARCFACE_MODEL_SHA256", "").strip()
    scrfd_path = Path(os.getenv("SCRFD_MODEL_PATH", "models/scrfd.onnx"))

    if not expected_sha:
        raise RuntimeError("ARCFACE_MODEL_SHA256 is required")

    registry = ModelRegistry({
        model_version: ModelSpec(
            model_id=os.getenv("FACE_MODEL_ID", "arcface_r50"),
            model_version=model_version,
            model_path=arcface_path,
            sha256=expected_sha,
            embedding_dim=512,
        )
    })
    spec = registry.validate(model_version)

    if not scrfd_path.is_file():
        raise FileNotFoundError(f"SCRFD model asset not found: {scrfd_path}")

    detector = SCRFDDetector(
        model_path=scrfd_path,
        conf_threshold=float(os.getenv("SCRFD_CONF_THRESHOLD", "0.5")),
        nms_threshold=float(os.getenv("SCRFD_NMS_THRESHOLD", "0.4")),
        input_size=(640, 640),
    )
    recognizer = ArcFaceRecognizer(
        model_path=spec.model_path,
        model_name=spec.model_id,
        model_version=spec.model_version,
    )
    if recognizer.embedding_dim != spec.embedding_dim:
        raise RuntimeError("Recognition model embedding contract mismatch")

    return RecognitionPipeline(
        detector=detector,
        recognizer=recognizer,
        aligner=FaceAligner(),
        quality_gate=FaceQualityGate(),
    )
