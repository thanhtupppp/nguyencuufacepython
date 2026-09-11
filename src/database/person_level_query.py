"""Compatibility builder for correctness-first person-level search SQL.

The canonical SQL is also exposed by :mod:`src.database.person_search`. This
small builder exists so tests and callers can inspect the query contract
without duplicating ranking semantics in application code.
"""

PERSON_LEVEL_SEARCH_SQL = """
SELECT person_id, similarity, name
FROM (
    SELECT
        e.person_id,
        1.0 - (e.embedding <=> %s::vector) AS similarity,
        p.name,
        ROW_NUMBER() OVER (
            PARTITION BY e.person_id
            ORDER BY e.embedding <=> %s::vector ASC
        ) AS rn
    FROM face_embeddings AS e
    JOIN persons AS p ON p.person_id = e.person_id
    WHERE e.model_version = %s
      AND (%s = FALSE OR p.status = 'active')
) AS ranked
WHERE rn = 1
ORDER BY similarity DESC
LIMIT %s;
"""


def build_person_level_search_sql(active_only: bool = True) -> str:
    """Return the canonical person-level ranking SQL contract.

    ``active_only`` is represented as a bound parameter rather than interpolated
    into SQL, preserving one statement shape and avoiding SQL construction from
    caller-controlled values.
    """
    # Keep the flag in the parameterized query; callers bind it at execution.
    # The argument is intentionally accepted to preserve the public contract.
    _ = active_only
    return PERSON_LEVEL_SEARCH_SQL


__all__ = ["PERSON_LEVEL_SEARCH_SQL", "build_person_level_search_sql"]
