"""
Abstract base class for face recognition feature extractors.
Ensures uniform interface across ArcFace, AdaFace, MagFace, MobileFaceNet, etc.
"""

from abc import ABC, abstractmethod
from typing import Union
import numpy as np


class BaseFaceRecognizer(ABC):
    """
    Abstract interface for face feature extractors.
    All subclasses must output L2-normalized embeddings.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name of the recognition model architecture (e.g. 'arcface_r50')."""
        pass

    @property
    @abstractmethod
    def model_version(self) -> str:
        """
        Version tag of the model (e.g. 'arcface_v1').
        Crucial for preventing cross-matching incompatible embeddings in database.
        """
        pass

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Dimension of feature vector (typically 512)."""
        pass

    @abstractmethod
    def extract_embedding(self, aligned_face_112: np.ndarray) -> np.ndarray:
        """
        Extract a normalized feature vector from a 112x112 aligned face crop.

        :param aligned_face_112: (112, 112, 3) BGR or RGB image
        :return: (512,) float32 vector with L2 norm = 1.0
        """
        pass

    def batch_extract(self, aligned_faces: list[np.ndarray]) -> np.ndarray:
        """
        Batch extraction of feature vectors.
        Default implementation iterates over list. Subclasses may optimize with batched tensor inference.
        """
        embeddings = [self.extract_embedding(f) for f in aligned_faces]
        return np.vstack(embeddings)

    @staticmethod
    def validate_embedding(
        embedding: np.ndarray,
        expected_dim: int = 512,
        norm_tolerance: float = 1e-3,
    ) -> np.ndarray:
        """Validate and normalize the persisted recognition embedding contract.

        Rejects malformed, non-finite, zero/near-zero, or wrong-dimensional vectors.
        Returns a float32 unit vector suitable for cosine similarity / pgvector.
        """
        array = np.asarray(embedding)
        if array.ndim not in (1, 2):
            raise ValueError("Embedding must be a 1D vector or 2D batch")
        if array.shape[-1] != expected_dim:
            raise ValueError(f"Expected {expected_dim}-D embedding, got {array.shape[-1]}-D")
        if not np.issubdtype(array.dtype, np.number):
            raise ValueError("Embedding must contain numeric values")
        array = array.astype(np.float32, copy=False)
        if not np.all(np.isfinite(array)):
            raise ValueError("Embedding contains NaN or Inf")
        if array.ndim == 1:
            norm = float(np.linalg.norm(array))
            if norm <= 1e-10:
                raise ValueError("Embedding norm is zero/near-zero")
            normalized = array / norm
        else:
            norms = np.linalg.norm(array, axis=1, keepdims=True)
            if np.any(norms <= 1e-10):
                raise ValueError("Batch contains zero/near-zero embedding")
            normalized = array / norms
        if not np.all(np.isfinite(normalized)):
            raise ValueError("Normalized embedding contains NaN or Inf")
        norms_after = np.linalg.norm(normalized, axis=-1)
        if not np.all(np.abs(norms_after - 1.0) <= norm_tolerance):
            raise ValueError("Embedding L2 normalization contract violated")
        return normalized.astype(np.float32, copy=False)

    @staticmethod
    def l2_normalize(embedding: np.ndarray, eps: float = 1e-10) -> np.ndarray:
        """
        Normalize 1D or 2D embedding array to unit length (L2 norm = 1.0).
        """
        if embedding.ndim == 1:
            norm = np.linalg.norm(embedding)
            return embedding / max(float(norm), eps)
        norm = np.linalg.norm(embedding, axis=1, keepdims=True)
        return embedding / np.maximum(norm, eps)
