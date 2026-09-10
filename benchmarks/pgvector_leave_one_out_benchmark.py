"""Leakage-safe person-level pgvector benchmark.

Unlike a self-query benchmark, each query embedding is excluded from the
candidate gallery. Exact person-level ranking is the ground truth; HNSW
retrieves raw-template candidates, then groups by person_id.

Required environment:
  DATABASE_URL=postgresql://...
  MODEL_VERSION=arcface_v1
  TOP_K=1
  HNSW_CANDIDATE_POOL=100
  HNSW_EF_SEARCH=100
  REPEATS=50

Optional:
  MAX_QUERIES=0                 # 0 = all eligible templates
  MIN_TEMPLATES_PER_PERSON=2
  ACTIVE_ONLY=1

A query person must retain at least one gallery template after excluding the
query template. No synthetic embeddings are generated.
"""

from __future__ import annotations

import os
import statistics
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import psycopg
from pgvector.psycopg import register_vector


EXACT_SQL = """
SELECT person_id, similarity
FROM (
    SELECT
        e.person_id,
        1.0 - (e.embedding <=> %s::vector) AS similarity,
        ROW_NUMBER() OVER (
            PARTITION BY e.person_id
            ORDER BY e.embedding <=> %s::vector ASC, e.id ASC
        ) AS rn
    FROM face_embeddings AS e
    JOIN persons AS p ON p.person_id = e.person_id
    WHERE e.model_version = %s
      AND e.id <> %s
      AND (%s = FALSE OR p.status = 'active')
) ranked
WHERE rn = 1
ORDER BY similarity DESC
LIMIT %s;
"""

HNSW_SQL = """
WITH candidates AS (
    SELECT
        e.person_id,
        1.0 - (e.embedding <=> %s::vector) AS similarity
    FROM face_embeddings AS e
    JOIN persons AS p ON p.person_id = e.person_id
    WHERE e.model_version = %s
      AND e.id <> %s
      AND (%s = FALSE OR p.status = 'active')
    ORDER BY e.embedding <=> %s::vector ASC
    LIMIT %s
), grouped AS (
    SELECT person_id, MAX(similarity) AS similarity
    FROM candidates
    GROUP BY person_id
)
SELECT person_id, similarity
FROM grouped
ORDER BY similarity DESC
LIMIT %s;
"""


@dataclass(frozen=True)
class Query:
    embedding_id: int
    person_id: str
    vector: np.ndarray


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(np.ceil(p / 100 * len(ordered))) - 1))
    return ordered[index]


def load_queries(conn: Any, model: str, min_templates: int, active_only: bool, max_queries: int) -> list[Query]:
    sql = """
    SELECT e.id, e.person_id, e.embedding
    FROM face_embeddings e
    JOIN persons p ON p.person_id = e.person_id
    WHERE e.model_version = %s
      AND (%s = FALSE OR p.status = 'active')
      AND (
          SELECT COUNT(*)
          FROM face_embeddings e2
          WHERE e2.person_id = e.person_id
            AND e2.model_version = e.model_version
      ) >= %s
    ORDER BY e.person_id, e.id;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (model, active_only, min_templates))
        rows = cur.fetchall()
    queries = []
    for embedding_id, person_id, embedding in rows:
        vector = np.asarray(embedding, dtype=np.float32)
        norm = np.linalg.norm(vector)
        if norm <= 1e-6:
            continue
        queries.append(Query(int(embedding_id), str(person_id), vector / norm))
        if max_queries and len(queries) >= max_queries:
            break
    return queries


def exact(conn: Any, query: Query, model: str, top_k: int, active_only: bool) -> tuple[str, ...]:
    with conn.cursor() as cur:
        cur.execute(EXACT_SQL, (query.vector, query.vector, model, query.embedding_id, active_only, top_k))
        return tuple(str(row[0]) for row in cur.fetchall())


def hnsw(conn: Any, query: Query, model: str, top_k: int, active_only: bool, candidate_pool: int) -> tuple[str, ...]:
    with conn.cursor() as cur:
        cur.execute(
            HNSW_SQL,
            (query.vector, model, query.embedding_id, active_only, query.vector, candidate_pool, top_k),
        )
        return tuple(str(row[0]) for row in cur.fetchall())


def main() -> None:
    database_url = os.environ["DATABASE_URL"]
    model = os.getenv("MODEL_VERSION", "arcface_v1")
    top_k = int(os.getenv("TOP_K", "1"))
    candidate_pool = int(os.getenv("HNSW_CANDIDATE_POOL", "100"))
    ef_search = int(os.getenv("HNSW_EF_SEARCH", "100"))
    repeats = int(os.getenv("REPEATS", "50"))
    min_templates = int(os.getenv("MIN_TEMPLATES_PER_PERSON", "2"))
    active_only = os.getenv("ACTIVE_ONLY", "1") not in {"0", "false", "False"}
    max_queries = int(os.getenv("MAX_QUERIES", "0"))

    if top_k < 1 or candidate_pool < top_k:
        raise ValueError("TOP_K must be >= 1 and HNSW_CANDIDATE_POOL must be >= TOP_K")

    with psycopg.connect(database_url) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector';")
            extension = cur.fetchone()
            if not extension:
                raise RuntimeError("pgvector extension is not installed")
            cur.execute("SELECT COUNT(*) FROM persons WHERE status = 'active';")
            active_people = int(cur.fetchone()[0])
            cur.execute("SELECT COUNT(*) FROM face_embeddings WHERE model_version = %s;", (model,))
            embeddings = int(cur.fetchone()[0])
            cur.execute("SET hnsw.ef_search = %s;", (ef_search,))

        queries = load_queries(conn, model, min_templates, active_only, max_queries)
        if not queries:
            raise RuntimeError("No leakage-safe queries found; need >= MIN_TEMPLATES_PER_PERSON templates/person")

        exact_results = {q.embedding_id: exact(conn, q, model, top_k, active_only) for q in queries}

        # Warmup is deliberately separate from measured samples.
        for q in queries[: min(20, len(queries))]:
            hnsw(conn, q, model, top_k, active_only, candidate_pool)

        latencies: list[float] = []
        matches = 0
        relevant_found = 0
        for q in queries:
            samples = []
            for _ in range(repeats):
                started = time.perf_counter_ns()
                result = hnsw(conn, q, model, top_k, active_only, candidate_pool)
                samples.append((time.perf_counter_ns() - started) / 1_000_000)
            latencies.extend(samples)
            ann = hnsw(conn, q, model, top_k, active_only, candidate_pool)
            exact_ids = exact_results[q.embedding_id]
            if ann == exact_ids:
                matches += 1
            if q.person_id in ann:
                relevant_found += 1

        print({
            "benchmark": "leave_one_out_person_level",
            "model_version": model,
            "pgvector_version": extension[0],
            "active_people": active_people,
            "embeddings": embeddings,
            "queries": len(queries),
            "min_templates_per_person": min_templates,
            "top_k": top_k,
            "hnsw_candidate_pool": candidate_pool,
            "hnsw_ef_search": ef_search,
            "exact_topk_agreement": matches / len(queries),
            "query_person_recall_at_k": relevant_found / len(queries),
            "hnsw_latency_ms_p50": percentile(latencies, 50),
            "hnsw_latency_ms_p95": percentile(latencies, 95),
            "hnsw_latency_ms_p99": percentile(latencies, 99),
            "hnsw_latency_ms_mean": statistics.mean(latencies) if latencies else 0.0,
        })


if __name__ == "__main__":
    main()
