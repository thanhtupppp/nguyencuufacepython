"""Runtime registry for versioned face-recognition model assets.

The registry deliberately fails closed: an embedding model is usable only when
its ONNX asset exists, its SHA-256 fingerprint matches the configured value,
its declared embedding contract matches the runtime recognizer, and provenance
/license metadata is complete and explicitly approved.
"""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    model_version: str
    model_path: Path
    sha256: str
    embedding_dim: int = 512
    publisher: str = ""
    upstream_revision: str = ""
    weight_license: str = ""
    commercial_use: str = ""
    provenance_url: str = ""


class ModelRegistry:
    def __init__(self, specs: Mapping[str, ModelSpec]):
        self._specs = dict(specs)

    def get(self, model_version: str) -> ModelSpec:
        try:
            return self._specs[model_version]
        except KeyError as exc:
            raise KeyError(f"Unknown recognition model_version: {model_version}") from exc

    @staticmethod
    def fingerprint(path: Path) -> str:
        digest = sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _require_metadata(spec: ModelSpec) -> None:
        fields = {
            "publisher": spec.publisher,
            "upstream_revision": spec.upstream_revision,
            "weight_license": spec.weight_license,
            "commercial_use": spec.commercial_use,
            "provenance_url": spec.provenance_url,
        }
        missing = [name for name, value in fields.items() if not str(value).strip()]
        if missing:
            raise ValueError(
                "Recognition model provenance/license metadata missing: "
                + ", ".join(sorted(missing))
            )
        if spec.commercial_use.strip().lower() not in {"approved", "allowed"}:
            raise ValueError(
                "Recognition model commercial_use must be explicitly approved/allowed"
            )

    def validate(self, model_version: str) -> ModelSpec:
        spec = self.get(model_version)
        self._require_metadata(spec)
        if not spec.sha256.strip() or spec.sha256.strip().lower().startswith("replace_with_"):
            raise ValueError(f"Verified SHA-256 is required for {model_version}")
        if not spec.model_path.is_file():
            raise FileNotFoundError(f"Model asset not found: {spec.model_path}")
        actual = self.fingerprint(spec.model_path)
        if actual.lower() != spec.sha256.lower():
            raise ValueError(
                f"Model fingerprint mismatch for {model_version}: expected {spec.sha256}, got {actual}"
            )
        if spec.embedding_dim != 512:
            raise ValueError("Current production contract requires 512-D embeddings")
        return spec

    def validate_detector_artifact(self, path: Path, expected_sha256: str) -> None:
        if not expected_sha256.strip() or expected_sha256.strip().lower().startswith("replace_with_"):
            raise ValueError("Verified SCRFD SHA-256 is required")
        if not path.is_file():
            raise FileNotFoundError(f"SCRFD model asset not found: {path}")
        actual = self.fingerprint(path)
        if actual.lower() != expected_sha256.lower():
            raise ValueError(
                f"SCRFD fingerprint mismatch: expected {expected_sha256}, got {actual}"
            )
