"""DatabaseClient variant that enforces person-level PostgreSQL ranking."""

import numpy as np

from .client import DatabaseClient as _BaseDatabaseClient, SearchCandidate
from .person_search import search_people


class DatabaseClient(_BaseDatabaseClient):
    """Database client with correctness-first person-level PostgreSQL search."""

    def search_top_k(self, query_vector: np.ndarray, model_version: str = "arcface_v1", top_k: int = 2, active_only: bool = True) -> list[SearchCandidate]:
        if self.use_sqlite or self.use_memory:
            return super().search_top_k(query_vector, model_version=model_version, top_k=top_k, active_only=active_only)
        rows = search_people(self._conn, query_vector=query_vector, model_version=model_version, top_k=top_k, active_only=active_only)
        return [SearchCandidate(person_id=pid, similarity=sim, name=name) for pid, sim, name in rows]


__all__ = ["DatabaseClient"]
