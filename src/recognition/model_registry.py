"""Runtime registry for versioned face-recognition model assets.

The registry deliberately fails closed: an embedding model is usable only when
its ONNX asset exists, its SHA-256 fingerprint matches the configured value,
and its declared embedding contract matches the runtime recognizer.
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

    def validate(self, model_version: str) -> ModelSpec:
        spec = self.get(model_version)
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
