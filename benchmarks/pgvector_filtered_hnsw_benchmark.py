"""Filtered HNSW iterative-scan benchmark.

This benchmark reuses the leakage-safe person-level setup but compares
pgvector HNSW iterative scan modes under filtering. It is intentionally
separate from the production search path until real measurements validate
correctness and latency.

Required:
  DATABASE_URL=postgresql://...
  MODEL_VERSION=arcface_v1

Optional:
  TOP_K=1
  CANDIDATE_POOL=100
  EF_SEARCH=100
  MAX_SCAN_TUPLES=20000
  ITERATIVE_SCAN=off|strict_order|relaxed_order
  REPEATS=20
  MIN_TEMPLATES_PER_PERSON=2
  ACTIVE_ONLY=1
  MAX_QUERIES=0

No synthetic embeddings are generated. The query template is excluded from
its gallery and the exact query is the correctness reference.
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
SELECT person_id
FROM (
    SELECT e.person_id,
           ROW_NUMBER() OVER (
               PARTITION BY e.person_id
               ORDER BY e.embedding <=> %s::vector ASC, e.id ASC
           ) AS rn
    FROM face_embeddings e
    JOIN persons p ON p.person_id = e.person_id
    WHERE e.model_version = %s
      AND e.id <> %s
      AND (%s = FALSE OR p.status = 'active')
) ranked
WHERE rn = 1
ORDER BY (SELECT MIN(e2.embedding <=> %s::vector)
          FROM face_embeddings e2
          WHERE e2.person_id = ranked.person_id
            AND e2.model_version = %s
            AND e2.id <> %s) ASC
LIMIT %s;
"""

HNSW_SQL = """
WITH candidates AS (
    SELECT e.person_id,
           e.embedding <=> %s::vector AS distance
    FROM face_embeddings e
    JOIN persons p ON p.person_id = e.person_id
    WHERE e.model_version = %s
      AND e.id <> %s
      AND (%s = FALSE OR p.status = 'active')
    ORDER BY e.embedding <=> %s::vector ASC
    LIMIT %s
), grouped AS (
    SELECT person_id, MIN(distance) AS distance
    FROM candidates
    GROUP BY person_id
)
SELECT person_id
FROM grouped
ORDER BY distance ASC
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
          SELECT COUNT(*) FROM face_embeddings e2
          WHERE e2.person_id = e.person_id
            AND e2.model_version = e.model_version
      ) >= %s
    ORDER BY e.person_id, e.id;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (model, active_only, min_templates))
        rows = cur.fetchall()
    result: list[Query] = []
    for embedding_id, person_id, embedding in rows:
        vector = np.asarray(embedding, dtype=np.float32)
        norm = np.linalg.norm(vector)
        if norm <= 1e-6:
            continue
        result.append(Query(int(embedding_id), str(person_id), vector / norm))
        if max_queries and len(result) >= max_queries:
            break
    return result


def exact(conn: Any, q: Query, model: str, top_k: int, active_only: bool) -> tuple[str, ...]:
    with conn.cursor() as cur:
        cur.execute(EXACT_SQL, (q.vector, model, q.embedding_id, active_only, q.vector, model, q.embedding_id, top_k))
        return tuple(str(row[0]) for row in cur.fetchall())


def hnsw(conn: Any, q: Query, model: str, top_k: int, active_only: bool, pool: int) -> tuple[str, ...]:
    with conn.cursor() as cur:
        cur.execute(HNSW_SQL, (q.vector, model, q.embedding_id, active_only, q.vector, pool, top_k))
        return tuple(str(row[0]) for row in cur.fetchall())


def main() -> None:
    database_url = os.environ["DATABASE_URL"]
    model = os.getenv("MODEL_VERSION", "arcface_v1")
    top_k = int(os.getenv("TOP_K", "1"))
    pool = int(os.getenv("CANDIDATE_POOL", "100"))
    ef_search = int(os.getenv("EF_SEARCH", "100"))
    max_scan = int(os.getenv("MAX_SCAN_TUPLES", "20000"))
    iterative = os.getenv("ITERATIVE_SCAN", "off")
    repeats = int(os.getenv("REPEATS", "20"))
    min_templates = int(os.getenv("MIN_TEMPLATES_PER_PERSON", "2"))
    active_only = os.getenv("ACTIVE_ONLY", "1") not in {"0", "false", "False"}
    max_queries = int(os.getenv("MAX_QUERIES", "0"))

    if iterative not in {"off", "strict_order", "relaxed_order"}:
        raise ValueError("ITERATIVE_SCAN must be off, strict_order, or relaxed_order")
    if top_k < 1 or pool < top_k:
        raise ValueError("TOP_K must be >= 1 and CANDIDATE_POOL must be >= TOP_K")

    with psycopg.connect(database_url) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector';")
            extension = cur.fetchone()
            if not extension:
                raise RuntimeError("pgvector extension is not installed")
            cur.execute("SET hnsw.ef_search = %s;", (ef_search,))
            cur.execute("SET hnsw.max_scan_tuples = %s;", (max_scan,))
            cur.execute("SET hnsw.iterative_scan = %s;", (iterative,))

        queries = load_queries(conn, model, min_templates, active_only, max_queries)
        if not queries:
            raise RuntimeError("No leakage-safe queries found")

        exact_results = {q.embedding_id: exact(conn, q, model, top_k, active_only) for q in queries}
        for q in queries[: min(20, len(queries))]:
            hnsw(conn, q, model, top_k, active_only, pool)

        latencies: list[float] = []
        agreement = 0
        recall = 0
        shortfall = 0
        for q in queries:
            last = ()
            for _ in range(repeats):
                started = time.perf_counter_ns()
                last = hnsw(conn, q, model, top_k, active_only, pool)
                latencies.append((time.perf_counter_ns() - started) / 1_000_000)
            if last == exact_results[q.embedding_id]:
                agreement += 1
            if q.person_id in last:
                recall += 1
            if len(last) < top_k:
                shortfall += 1

        print({
            "benchmark": "filtered_hnsw_iterative_scan",
            "model_version": model,
            "pgvector_version": extension[0],
            "iterative_scan": iterative,
            "hnsw_ef_search": ef_search,
            "hnsw_max_scan_tuples": max_scan,
            "candidate_pool": pool,
            "queries": len(queries),
            "top_k": top_k,
            "exact_topk_agreement": agreement / len(queries),
            "query_person_recall_at_k": recall / len(queries),
            "shortfall_rate": shortfall / len(queries),
            "latency_ms_p50": percentile(latencies, 50),
            "latency_ms_p95": percentile(latencies, 95),
            "latency_ms_p99": percentile(latencies, 99),
            "latency_ms_mean": statistics.mean(latencies) if latencies else 0.0,
        })


if __name__ == "__main__":
    main()
