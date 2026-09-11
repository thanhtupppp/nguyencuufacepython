import hashlib
import json

import pytest

from src.recognition.model_provenance import load_manifest, verify_artifact, verify_sha256


def test_verify_sha256(tmp_path):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"locked-model")
    expected = hashlib.sha256(b"locked-model").hexdigest()
    assert verify_sha256(model, expected) == expected


def test_verify_sha256_rejects_mismatch(tmp_path):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"locked-model")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_sha256(model, "0" * 64)


def test_manifest_requires_provenance_fields(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"model_id": "arcface_r50"}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing required fields"):
        load_manifest(manifest)


def test_verify_artifact_requires_file(tmp_path):
    artifact = tmp_path / "missing.onnx"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "model_id": "arcface_r50",
        "version": "v1",
        "path": str(artifact),
        "sha256": "0" * 64,
        "license": "MIT",
        "source": "https://example.invalid/model"
    }), encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        verify_artifact(load_manifest(manifest))
