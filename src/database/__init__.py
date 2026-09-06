"""
Database subpackage for vector search and identity management.
"""

from .client import DatabaseClient, SearchCandidate, RecognitionDecision

__all__ = ["DatabaseClient", "SearchCandidate", "RecognitionDecision"]
