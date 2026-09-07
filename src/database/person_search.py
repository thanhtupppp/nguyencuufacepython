"""PostgreSQL person-level vector search helpers.

This module keeps the correctness-critical ranking semantics in one place so
callers cannot accidentally reintroduce raw embedding LIMIT + application
side de-duplication.
"""

from typing import Any

import numpy as np


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


def search_people(
    conn: Any,
    query_vector: np.ndarray,
    model_version: str,
    top_k: int,
    active_only: bool = True,
) -> list[tuple[str, float, str | None]]:
    """Return at most ``top_k`` distinct people ranked by best template.

    Correctness invariant: LIMIT is applied only after the best template for
    every eligible person has been selected. This makes top1/top2 a valid
    person-level margin even when enrollment template counts differ.
    """
    if top_k < 1:
        raise ValueError("top_k must be >= 1")

    q = np.asarray(query_vector, dtype=np.float32).flatten()
    norm = np.linalg.norm(q)
    if norm <= 1e-6:
        raise ValueError("query_vector must have non-zero norm")
    q = q / norm

    with conn.cursor() as cur:
        cur.execute(
            PERSON_LEVEL_SEARCH_SQL,
            (q, q, model_version, active_only, top_k),
        )
        return [
            (str(person_id), float(similarity), name)
            for person_id, similarity, name in cur.fetchall()
        ]
