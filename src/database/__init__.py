"""
Database subpackage for vector search and identity management.
"""

from .person_level_client import DatabaseClient
from .client import SearchCandidate, RecognitionDecision

__all__ = ["DatabaseClient", "SearchCandidate", "RecognitionDecision"]
