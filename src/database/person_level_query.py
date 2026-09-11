"""Compatibility builder for correctness-first person-level search SQL."""

_BASE_SQL = """
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
"""

_SUFFIX_SQL = """
) AS ranked
WHERE rn = 1
ORDER BY similarity DESC
LIMIT %s;
"""


def build_person_level_search_sql(active_only: bool = True) -> str:
    """Return parameterized person-level ranking SQL.

    LIMIT is applied only after the best template per person has been selected.
    The active filter is included only when requested; its value remains a
    bound parameter at execution time.
    """
    if active_only:
        return _BASE_SQL + "      AND p.status = 'active'\n" + _SUFFIX_SQL
    return _BASE_SQL + _SUFFIX_SQL


PERSON_LEVEL_SEARCH_SQL = build_person_level_search_sql(True)

__all__ = ["PERSON_LEVEL_SEARCH_SQL", "build_person_level_search_sql"]
