"""Build the fail-closed recognition runtime for API processes."""

import os
from pathlib import Path

from src.alignment.aligner import FaceAligner
from src.detection.scrfd import SCRFDDetector
from src.quality.quality_gate import FaceQualityGate
from src.anti_spoofing.liveness import AntiSpoofDetector
from src.recognition.arcface import ArcFaceRecognizer
from src.recognition.model_registry import ModelRegistry, ModelSpec
from src.recognition.pipeline import RecognitionPipeline


def build_recognition_pipeline() -> RecognitionPipeline:
    """Create the production pipeline from environment configuration.

    No recognition model asset or fingerprint means no API recognition service
    is created. If LIVENESS_MODEL_PATH is configured, liveness is mandatory and
    strict: missing/invalid weights never fall back to the Fourier heuristic.
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

    liveness = None
    liveness_path = os.getenv("LIVENESS_MODEL_PATH", "").strip()
    if liveness_path:
        path = Path(liveness_path)
        if not path.is_file():
            raise FileNotFoundError(f"Liveness model asset not found: {path}")
        liveness = AntiSpoofDetector(
            model_path=path,
            threshold=float(os.getenv("LIVENESS_THRESHOLD", "0.85")),
            strict_mode=True,
            min_face_size=int(os.getenv("LIVENESS_MIN_FACE_SIZE", "60")),
        )

    return RecognitionPipeline(
        detector=detector,
        recognizer=recognizer,
        aligner=FaceAligner(),
        quality_gate=FaceQualityGate(),
        liveness=liveness,
    )
