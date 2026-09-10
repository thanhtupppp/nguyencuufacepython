"""Correctness-first PostgreSQL person-level vector search.

This module keeps the recognition invariant explicit: LIMIT is applied only
after the best embedding for each eligible person_id has been selected.
It is intentionally separate from DatabaseClient until the real pgvector
benchmark validates the query plan and ANN trade-offs.
"""

from __future__ import annotations

PERSON_LEVEL_SEARCH_SQL = """
SELECT person_id, similarity, name
FROM (
    SELECT
        e.person_id,
        1.0 - (e.embedding <=> %s::vector) AS similarity,
        p.name,
        ROW_NUMBER() OVER (
            PARTITION BY e.person_id
            ORDER BY e.embedding <=> %s::vector ASC, e.id ASC
        ) AS rn
    FROM face_embeddings AS e
    JOIN persons AS p ON p.person_id = e.person_id
    WHERE e.model_version = %s
      {status_filter}
) ranked
WHERE rn = 1
ORDER BY similarity DESC
LIMIT %s;
"""


def build_person_level_search_sql(active_only: bool = True) -> str:
    """Return the exact person-level query without changing ANN behavior."""
    status_filter = "AND p.status = 'active'" if active_only else ""
    return PERSON_LEVEL_SEARCH_SQL.format(status_filter=status_filter)
