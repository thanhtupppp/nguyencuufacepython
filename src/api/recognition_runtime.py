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


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is required for verified recognition runtime")
    return value


def build_recognition_pipeline() -> RecognitionPipeline:
    """Create the production pipeline only from explicitly approved assets.

    Both ArcFace and SCRFD are provenance-gated. Missing SHA/license/provenance
    metadata, missing files, or fingerprint mismatches prevent initialization.
    """
    model_version = os.getenv("FACE_MODEL_VERSION", "arcface_v1")
    arcface_path = Path(os.getenv("ARCFACE_MODEL_PATH", "models/arcface.onnx"))
    scrfd_path = Path(os.getenv("SCRFD_MODEL_PATH", "models/scrfd.onnx"))

    registry = ModelRegistry({
        model_version: ModelSpec(
            model_id=os.getenv("FACE_MODEL_ID", "arcface_r50"),
            model_version=model_version,
            model_path=arcface_path,
            sha256=_required_env("ARCFACE_MODEL_SHA256"),
            embedding_dim=512,
            publisher=_required_env("ARCFACE_MODEL_PUBLISHER"),
            upstream_revision=_required_env("ARCFACE_MODEL_REVISION"),
            weight_license=_required_env("ARCFACE_MODEL_WEIGHT_LICENSE"),
            commercial_use=_required_env("ARCFACE_MODEL_COMMERCIAL_USE"),
            provenance_url=_required_env("ARCFACE_MODEL_PROVENANCE_URL"),
        )
    })
    spec = registry.validate(model_version)

    registry.validate_detector_artifact(
        scrfd_path,
        _required_env("SCRFD_MODEL_SHA256"),
    )
    _required_env("SCRFD_MODEL_PUBLISHER")
    _required_env("SCRFD_MODEL_REVISION")
    _required_env("SCRFD_MODEL_WEIGHT_LICENSE")
    scrfd_commercial_use = _required_env("SCRFD_MODEL_COMMERCIAL_USE").lower()
    if scrfd_commercial_use not in {"approved", "allowed"}:
        raise ValueError("SCRFD_MODEL_COMMERCIAL_USE must be explicitly approved/allowed")
    _required_env("SCRFD_MODEL_PROVENANCE_URL")

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
