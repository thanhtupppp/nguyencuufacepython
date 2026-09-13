from pathlib import Path
import hashlib

import pytest

from src.recognition.model_registry import ModelRegistry, ModelSpec


def spec(path: Path, sha: str) -> ModelSpec:
    return ModelSpec(
        "arcface_r50", "arcface_v1", path, sha,
        publisher="test-publisher",
        upstream_revision="test-revision",
        weight_license="test-license",
        commercial_use="approved",
        provenance_url="https://example.invalid/model",
    )


def test_registry_rejects_missing_asset(tmp_path: Path):
    registry = ModelRegistry({"arcface_v1": spec(tmp_path / "missing.onnx", "0" * 64)})
    with pytest.raises(FileNotFoundError):
        registry.validate("arcface_v1")


def test_registry_rejects_unknown_version(tmp_path: Path):
    registry = ModelRegistry({})
    with pytest.raises(KeyError):
        registry.validate("arcface_v1")


def test_registry_accepts_matching_fingerprint(tmp_path: Path):
    asset = tmp_path / "model.onnx"
    asset.write_bytes(b"test-model")
    expected = hashlib.sha256(b"test-model").hexdigest()
    registry = ModelRegistry({"arcface_v1": spec(asset, expected)})
    assert registry.validate("arcface_v1").model_id == "arcface_r50"


def test_registry_rejects_tampered_asset(tmp_path: Path):
    asset = tmp_path / "model.onnx"
    asset.write_bytes(b"original")
    registry = ModelRegistry({"arcface_v1": spec(asset, "0" * 64)})
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        registry.validate("arcface_v1")
