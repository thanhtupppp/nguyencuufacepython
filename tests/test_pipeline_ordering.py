import numpy as np
import pytest

from src.recognition.pipeline import RecognitionPipeline


class Recorder:
    def __init__(self):
        self.calls = []


class Detector:
    def __init__(self, rec):
        self.rec = rec

    def detect(self, image):
        self.rec.calls.append("detection")
        return [{
            "bbox": [10, 10, 122, 122],
            "score": 0.99,
            "landmarks": np.array([[40, 45], [75, 45], [57, 66], [42, 88], [72, 88]], dtype=np.float32),
        }]


class Aligner:
    def __init__(self, rec):
        self.rec = rec

    def align(self, image, landmarks):
        self.rec.calls.append("alignment")
        return np.zeros((112, 112, 3), dtype=np.uint8), landmarks


class Recognizer:
    model_version = "arcface_test"
    embedding_dim = 512

    def __init__(self, rec):
        self.rec = rec

    def extract_embedding(self, aligned):
        self.rec.calls.append("embedding")
        vec = np.zeros(512, dtype=np.float32)
        vec[0] = 1.0
        return vec


class Quality:
    is_valid = True
    overall_quality_score = 0.95
    rejection_reasons = []


class FAS:
    def __init__(self, rec, decision="PASS"):
        self.rec = rec
        self.decision = decision

    def predict_liveness(self, image, bbox):
        self.rec.calls.append("fas")
        return type("Live", (), {
            "decision": type("Decision", (), {"value": self.decision})(),
            "liveness_score": 0.99,
            "attack_type": "SPOOF_DETECTED" if self.decision == "FAIL" else None,
            "reason": "MODEL_WEIGHTS_MISSING" if self.decision != "PASS" else None,
        })()


def _pipeline(rec, fas):
    pipeline = RecognitionPipeline(Detector(rec), Recognizer(rec), Aligner(rec), liveness=fas)
    pipeline.quality_gate.assess_quality = lambda *args: (rec.calls.append("quality") or Quality())
    return pipeline


def test_pipeline_enforces_biometric_gate_order():
    rec = Recorder()
    result = _pipeline(rec, FAS(rec)).extract_best_face(np.zeros((160, 160, 3), dtype=np.uint8))
    assert result.embedding.shape == (512,)
    assert rec.calls == ["detection", "quality", "fas", "alignment", "embedding"]


def test_strict_pipeline_rejects_fas_before_embedding():
    rec = Recorder()
    with pytest.raises(ValueError, match="SPOOF_DETECTED|MODEL_WEIGHTS_MISSING"):
        _pipeline(rec, FAS(rec, decision="FAIL")).extract_best_face(
            np.zeros((160, 160, 3), dtype=np.uint8)
        )
    assert rec.calls == ["detection", "quality", "fas"]
    assert "embedding" not in rec.calls


class AuthorizationGate:
    def __init__(self, rec): self.rec = rec
    def authorize(self): self.rec.calls.append("authorization"); return True


class DecisionService:
    def __init__(self, rec): self.rec = rec
    def decide(self): self.rec.calls.append("decision"); return True


class Ledger:
    def __init__(self, rec, fail=False): self.rec, self.fail = rec, fail
    def record(self):
        self.rec.calls.append("ledger")
        if self.fail:
            raise RuntimeError("ledger unavailable")
        return True


class ActionTransport:
    def __init__(self, rec): self.rec = rec
    def send(self): self.rec.calls.append("action")


def execute_authorized_action(rec, *, ledger_fail=False):
    AuthorizationGate(rec).authorize()
    DecisionService(rec).decide()
    Ledger(rec, fail=ledger_fail).record()
    ActionTransport(rec).send()


def test_durable_ledger_must_precede_side_effect():
    rec = Recorder()
    execute_authorized_action(rec)
    assert rec.calls == ["authorization", "decision", "ledger", "action"]


def test_ledger_failure_blocks_side_effect():
    rec = Recorder()
    with pytest.raises(RuntimeError, match="ledger unavailable"):
        execute_authorized_action(rec, ledger_fail=True)
    assert rec.calls == ["authorization", "decision", "ledger"]
    assert "action" not in rec.calls
