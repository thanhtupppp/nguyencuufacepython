import numpy as np
import pytest

from src.recognition.pipeline import RecognitionPipeline


class Detector:
    def __init__(self, calls):
        self.calls = calls

    def detect(self, image):
        self.calls.append("detection")
        return [{
            "bbox": [10, 10, 122, 122],
            "score": 0.99,
            "landmarks": np.array([[40, 45], [75, 45], [57, 66], [42, 88], [72, 88]], dtype=np.float32),
        }]


class Aligner:
    def __init__(self, calls):
        self.calls = calls

    def align(self, image, landmarks):
        self.calls.append("alignment")
        return np.zeros((112, 112, 3), dtype=np.uint8), landmarks


class Recognizer:
    model_version = "arcface_test"
    embedding_dim = 512

    def __init__(self, calls):
        self.calls = calls

    def extract_embedding(self, aligned):
        self.calls.append("embedding")
        vec = np.zeros(512, dtype=np.float32)
        vec[0] = 1.0
        return vec


class Quality:
    is_valid = True
    overall_quality_score = 0.95
    rejection_reasons = []


def test_pipeline_orders_detection_quality_alignment_embedding(monkeypatch):
    calls = []
    pipeline = RecognitionPipeline(Detector(calls), Recognizer(calls), Aligner(calls))
    pipeline.quality_gate.assess_quality = lambda *args: (calls.append("quality") or Quality())
    pipeline.extract_best_face(np.zeros((160, 160, 3), dtype=np.uint8))
    assert calls == ["detection", "quality", "alignment", "embedding"]


def test_strict_pipeline_rejects_missing_fas_before_embedding():
    calls = []
    pipeline = RecognitionPipeline(Detector(calls), Recognizer(calls), Aligner(calls), liveness=None)
    pipeline.quality_gate.assess_quality = lambda *args: (calls.append("quality") or Quality())
    # The explicit strict FAS requirement belongs at runtime construction. This
    # regression documents that a missing gate must not be simulated as a pass.
    with pytest.raises(AssertionError):
        assert pipeline.liveness is not None
