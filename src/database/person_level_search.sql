-- Correctness-first PostgreSQL/pgvector person-level search.
--
-- IMPORTANT: this query intentionally ranks the best template per person
-- before applying LIMIT. Do not replace it with raw embedding LIMIT +
-- application-side deduplication; that can hide the second-best person and
-- corrupt the top1-top2 recognition margin when one person has many templates.
--
-- Parameters:
--   $1 = query vector
--   $2 = query vector (used by the window ordering)
--   $3 = model_version
--   $4 = top_k
--   $5 = active_only boolean

SELECT person_id, similarity, name
FROM (
    SELECT
        e.person_id,
        1.0 - (e.embedding <=> $1::vector) AS similarity,
        p.name,
        ROW_NUMBER() OVER (
            PARTITION BY e.person_id
            ORDER BY e.embedding <=> $2::vector ASC
        ) AS rn
    FROM face_embeddings AS e
    JOIN persons AS p ON p.person_id = e.person_id
    WHERE e.model_version = $3
      AND ($5 = FALSE OR p.status = 'active')
) AS ranked
WHERE rn = 1
ORDER BY similarity DESC
LIMIT $4;
