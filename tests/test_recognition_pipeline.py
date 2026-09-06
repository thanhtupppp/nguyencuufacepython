import numpy as np
import pytest

from src.recognition.pipeline import RecognitionPipeline


class FakeDetector:
    def detect(self, image):
        return [{
            "bbox": [10, 10, 122, 122],
            "score": 0.99,
            "landmarks": np.array([[40, 45], [75, 45], [57, 66], [42, 88], [72, 88]], dtype=np.float32),
        }]


class FakeRecognizer:
    model_version = "arcface_test"
    embedding_dim = 512

    def extract_embedding(self, aligned):
        vec = np.zeros(512, dtype=np.float32)
        vec[0] = 1.0
        return vec


def test_pipeline_enforces_end_to_end_contract(monkeypatch):
    pipeline = RecognitionPipeline(FakeDetector(), FakeRecognizer())
    monkeypatch.setattr(
        pipeline.quality_gate,
        "assess_quality",
        lambda image, bbox, landmarks: type("Q", (), {"is_valid": True, "overall_quality_score": 0.91})(),
    )
    result = pipeline.extract_best_face(np.zeros((160, 160, 3), dtype=np.uint8))
    assert result.embedding.shape == (512,)
    assert result.model_version == "arcface_test"
    assert result.quality_score == 0.91


def test_pipeline_rejects_no_face():
    class EmptyDetector:
        def detect(self, image):
            return []

    with pytest.raises(ValueError, match="NO_FACE_DETECTED"):
        RecognitionPipeline(EmptyDetector(), FakeRecognizer()).extract_best_face(
            np.zeros((160, 160, 3), dtype=np.uint8)
        )


def test_pipeline_rejects_quality_failures():
    class BadQuality:
        def detect(self, image):
            return [{
                "bbox": [10, 10, 122, 122],
                "score": 0.99,
                "landmarks": np.array([[40, 45], [75, 45], [57, 66], [42, 88], [72, 88]], dtype=np.float32),
            }]

    pipeline = RecognitionPipeline(BadQuality(), FakeRecognizer())
    monkey = lambda *args, **kwargs: type("Q", (), {"is_valid": False, "overall_quality_score": 0.1})()
    pipeline.quality_gate.assess_quality = monkey
    with pytest.raises(ValueError, match="NO_FACE_PASSED_QUALITY_GATE"):
        pipeline.extract_best_face(np.zeros((160, 160, 3), dtype=np.uint8))
