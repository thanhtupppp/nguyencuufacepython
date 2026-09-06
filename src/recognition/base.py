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

        :param aligned_faces: list of (112, 112, 3) numpy arrays
        :return: (N, embedding_dim) float32 array
        """
        embeddings = [self.extract_embedding(f) for f in aligned_faces]
        return np.vstack(embeddings)

    @staticmethod
    def l2_normalize(embedding: np.ndarray, eps: float = 1e-10) -> np.ndarray:
        """
        Normalize 1D or 2D embedding array to unit length (L2 norm = 1.0).
        """
        if embedding.ndim == 1:
            norm = np.linalg.norm(embedding)
            return embedding / max(norm, eps)
        norm = np.linalg.norm(embedding, axis=1, keepdims=True)
        return embedding / np.maximum(norm, eps)
