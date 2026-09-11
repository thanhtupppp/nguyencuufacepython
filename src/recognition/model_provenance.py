"""Fail-closed model artifact provenance and integrity verification."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelArtifact:
    model_id: str
    version: str
    path: Path
    sha256: str
    license: str
    source: str


def load_manifest(path: str | Path) -> ModelArtifact:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    required = ("model_id", "version", "path", "sha256", "license", "source")
    missing = [key for key in required if not data.get(key)]
    if missing:
        raise ValueError(f"Model manifest missing required fields: {', '.join(missing)}")
    return ModelArtifact(
        model_id=str(data["model_id"]),
        version=str(data["version"]),
        path=Path(data["path"]),
        sha256=str(data["sha256"]),
        license=str(data["license"]),
        source=str(data["source"]),
    )


def verify_sha256(path: str | Path, expected_sha256: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual.lower() != expected_sha256.lower():
        raise ValueError(
            f"Model SHA-256 mismatch: expected {expected_sha256}, got {actual}"
        )
    return actual


def verify_artifact(artifact: ModelArtifact) -> ModelArtifact:
    if not artifact.path.is_file():
        raise FileNotFoundError(f"Model artifact not found: {artifact.path}")
    verify_sha256(artifact.path, artifact.sha256)
    return artifact
