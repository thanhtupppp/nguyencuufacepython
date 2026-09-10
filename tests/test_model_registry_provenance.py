from hashlib import sha256
from pathlib import Path

import pytest

from src.recognition.model_registry import ModelRegistry, ModelSpec


def _spec(path: Path, **overrides) -> ModelSpec:
    data = dict(
        model_id="arcface_test",
        model_version="v1",
        model_path=path,
        sha256=sha256(path.read_bytes()).hexdigest(),
        embedding_dim=512,
        publisher="test-publisher",
        upstream_revision="test-revision",
        weight_license="approved-license",
        commercial_use="approved",
        provenance_url="https://example.invalid/model",
    )
    data.update(overrides)
    return ModelSpec(**data)


def test_registry_accepts_verified_asset(tmp_path: Path):
    path = tmp_path / "model.onnx"
    path.write_bytes(b"verified-model")
    spec = _spec(path)
    assert ModelRegistry({"v1": spec}).validate("v1") == spec


@pytest.mark.parametrize("field", ["publisher", "upstream_revision", "weight_license", "commercial_use", "provenance_url"])
def test_registry_rejects_missing_provenance(field: str, tmp_path: Path):
    path = tmp_path / "model.onnx"
    path.write_bytes(b"model")
    spec = _spec(path, **{field: ""})
    with pytest.raises(ValueError, match="provenance/license"):
        ModelRegistry({"v1": spec}).validate("v1")


def test_registry_rejects_unapproved_commercial_use(tmp_path: Path):
    path = tmp_path / "model.onnx"
    path.write_bytes(b"model")
    spec = _spec(path, commercial_use="research-only")
    with pytest.raises(ValueError, match="commercial_use"):
        ModelRegistry({"v1": spec}).validate("v1")


def test_registry_rejects_placeholder_sha(tmp_path: Path):
    path = tmp_path / "model.onnx"
    path.write_bytes(b"model")
    spec = _spec(path, sha256="REPLACE_WITH_VERIFIED_SHA256")
    with pytest.raises(ValueError, match="SHA-256"):
        ModelRegistry({"v1": spec}).validate("v1")


def test_registry_rejects_modified_asset(tmp_path: Path):
    path = tmp_path / "model.onnx"
    path.write_bytes(b"model")
    spec = _spec(path)
    path.write_bytes(b"modified-model")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        ModelRegistry({"v1": spec}).validate("v1")


def test_registry_rejects_non_512_embedding_contract(tmp_path: Path):
    path = tmp_path / "model.onnx"
    path.write_bytes(b"model")
    spec = _spec(path, embedding_dim=256)
    with pytest.raises(ValueError, match="512-D"):
        ModelRegistry({"v1": spec}).validate("v1")


def test_registry_rejects_missing_scrfd_sha(tmp_path: Path):
    path = tmp_path / "scrfd.onnx"
    path.write_bytes(b"scrfd")
    with pytest.raises(ValueError, match="SCRFD SHA-256"):
        ModelRegistry({}).validate_detector_artifact(path, "REPLACE_WITH_VERIFIED_SHA256")
