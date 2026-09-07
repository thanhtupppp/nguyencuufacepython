"""
ArcFace recognition module using ONNX Runtime.
Outputs 512-dimensional L2-normalized feature vectors.
"""

import hashlib
import os
from pathlib import Path
from typing import Any, Optional, Union
import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None

from .base import BaseFaceRecognizer


class ArcFaceRecognizer(BaseFaceRecognizer):
    """
    ArcFace (Additive Angular Margin Loss) feature extractor running on ONNX Runtime.
    Supports ResNet-50 / ResNet-100 backbones (e.g., glint360k, ms1mv2, buffalo_l w600k_r50).
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        model_name: str = "arcface_r50",
        model_version: str = "arcface_v1",
        providers: Optional[list[str]] = None,
        expected_sha256: Optional[str] = None,
    ):
        self._model_name = model_name
        self._model_version = model_version
        self._embedding_dim = 512
        self.model_path = Path(model_path) if model_path else None
        self.expected_sha256 = expected_sha256
        self.session: Any = None

        if providers is None:
            # Prefer CUDA/DirectML GPU if available, fallback to CPU
            available = ort.get_available_providers() if ort else []
            self.providers = [
                p
                for p in ["CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]
                if p in available
            ]
            if not self.providers and ort:
                self.providers = ["CPUExecutionProvider"]
        else:
            self.providers = providers

        if self.model_path and self.model_path.exists():
            if self.expected_sha256:
                self._verify_sha256(self.model_path, self.expected_sha256)
            self._load_model()

    @staticmethod
    def _verify_sha256(path: Path, expected: str) -> None:
        """Verifies model file checksum against expected SHA-256 hash."""
        digest = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        actual = digest.hexdigest()
        if actual.lower() != expected.lower():
            raise ValueError(
                f"Model SHA-256 fingerprint mismatch for {path}: expected {expected}, got {actual}"
            )

    def _load_model(self) -> None:
        """Initializes ONNX Runtime session and validates tensor contracts."""
        if ort is None:
            raise ImportError("onnxruntime is required to run ArcFaceRecognizer")

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Cap CPU threads to prevent thrashing high-core CPUs (e.g. 48 cores)
        session_options.intra_op_num_threads = min(4, os.cpu_count() or 4)
        session_options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(self.model_path), sess_options=session_options, providers=self.providers
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # Validate input & output tensor shape contracts
        in_shape = self.session.get_inputs()[0].shape
        if len(in_shape) == 4 and in_shape[2] != -1 and (in_shape[2], in_shape[3]) != (112, 112):
            raise ValueError(f"Expected 112x112 input tensor, got {in_shape}")

        out_shape = self.session.get_outputs()[0].shape
        if len(out_shape) == 2 and out_shape[1] != -1 and out_shape[1] != self._embedding_dim:
            raise ValueError(f"Expected {self._embedding_dim}-D output embedding, got {out_shape}")

        self.input_size = (112, 112)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def preprocess(self, aligned_face_112: np.ndarray) -> np.ndarray:
        """
        Preprocesses 112x112 face image to tensor:
        - Resize to (112, 112) if needed
        - Convert BGR to RGB
        - Normalize to [-1.0, 1.0] via (pixel - 127.5) / 127.5
        - Transpose to (1, 3, 112, 112) NCHW
        """
        if aligned_face_112.shape[:2] != (112, 112):
            aligned_face_112 = cv2.resize(aligned_face_112, (112, 112), interpolation=cv2.INTER_LINEAR)

        # Standard ArcFace expects RGB
        img_rgb = cv2.cvtColor(aligned_face_112, cv2.COLOR_BGR2RGB)
        tensor = (img_rgb.astype(np.float32) - 127.5) / 127.5
        tensor = np.transpose(tensor, (2, 0, 1))  # (3, 112, 112)
        tensor = np.expand_dims(tensor, axis=0)   # (1, 3, 112, 112)
        return tensor

    def extract_embedding(self, aligned_face_112: np.ndarray) -> np.ndarray:
        """
        Extracts L2-normalized 512D feature vector.

        :param aligned_face_112: 112x112 aligned face crop (BGR)
        :return: (512,) float32 normalized vector
        """
        if self.session is None:
            raise RuntimeError(
                f"Model session not initialized. Model path '{self.model_path}' not loaded."
            )

        tensor = self.preprocess(aligned_face_112)
        raw_output = self.session.run([self.output_name], {self.input_name: tensor})[0]
        embedding = raw_output.flatten()
        return self.l2_normalize(embedding)

    def batch_extract(self, aligned_faces: list[np.ndarray]) -> np.ndarray:
        """
        Batched extraction for high-throughput evaluation.
        """
        if not aligned_faces:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        if self.session is None:
            raise RuntimeError("Model session not initialized.")

        tensors = [self.preprocess(f) for f in aligned_faces]
        batch_tensor = np.vstack(tensors)  # (N, 3, 112, 112)

        raw_output = self.session.run([self.output_name], {self.input_name: batch_tensor})[0]
        return self.l2_normalize(raw_output)
