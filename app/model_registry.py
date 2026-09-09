"""Fail-closed registry boundary for verified SCRFD/ArcFace artifacts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

REQUIRED = (
    "model_id",
    "publisher",
    "upstream_revision",
    "artifact",
    "sha256",
    "weight_license",
    "commercial_use",
)
PLACEHOLDERS = {"", "replace_with_verified_sha256", "pending", "unknown", "tbd"}


class ModelRegistryError(RuntimeError):
    """Raised when a recognition model cannot be approved."""


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    entries = data.get("models", data) if isinstance(data, dict) else data
    if isinstance(entries, dict):
        entries = list(entries.values())
    if not isinstance(entries, list):
        raise ModelRegistryError("manifest_models_must_be_list")
    return entries


def verify_manifest(path: str | Path) -> dict[str, Any]:
    manifest = Path(path)
    if not manifest.is_file():
        raise ModelRegistryError(f"manifest_missing:{manifest}")
    failures: dict[str, list[str]] = {}
    for entry in _load_manifest(manifest):
        model_id = str(entry.get("model_id", "unknown"))
        errors = [f"missing:{k}" for k in REQUIRED if k not in entry]
        sha = str(entry.get("sha256", "")).strip().lower()
        if sha in PLACEHOLDERS or len(sha) != 64:
            errors.append("unverified_sha256")
        artifact = manifest.parent / str(entry.get("artifact", ""))
        if not artifact.is_file():
            errors.append("artifact_missing")
        license_value = str(entry.get("weight_license", "")).strip()
        if not license_value or license_value.lower() in {"unknown", "tbd", "pending"}:
            errors.append("weight_license_unverified")
        if len(sha) == 64 and artifact.is_file():
            actual = hashlib.sha256(artifact.read_bytes()).hexdigest()
            if actual != sha:
                errors.append(f"sha256_mismatch:{actual}")
        if errors:
            failures[model_id] = errors
    receipt = {"manifest": str(manifest), "verified": not failures, "failures": failures}
    if failures:
        raise ModelRegistryError(json.dumps(receipt, sort_keys=True))
    return receipt


class RecognitionModelRegistry:
    """Registry that refuses to expose unverified recognition artifacts."""

    def __init__(self, manifest_path: str | Path):
        self.manifest_path = Path(manifest_path)
        self._approved = False
        self._receipt: dict[str, Any] | None = None

    def approve(self) -> dict[str, Any]:
        self._receipt = verify_manifest(self.manifest_path)
        self._approved = True
        return self._receipt

    @property
    def approved(self) -> bool:
        return self._approved

    @property
    def receipt(self) -> dict[str, Any] | None:
        return self._receipt

    def require_approved(self) -> None:
        if not self._approved:
            raise ModelRegistryError("recognition_model_not_approved")
