"""Benchmark person-level exact vs HNSW pgvector retrieval.

The benchmark uses exact person-level ranking as ground truth. HNSW first
retrieves a configurable raw-embedding candidate pool using the pgvector
cosine index, then groups those candidates by person_id. Recall is measured
against exact person-level results. No synthetic embeddings are generated.

Environment:
  DATABASE_URL=postgresql://...
  MODEL_VERSION=arcface_v1
  TOP_K=1
  HNSW_CANDIDATE_POOL=100
  HNSW_EF_SEARCH=100
  WARMUP=20
  REPEATS=100
  QUERY_PERSON_IDS=person_001,person_002,... (optional)
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
            ORDER BY e.embedding <=> %s::vector ASC
        ) AS rn
    FROM face_embeddings AS e
    JOIN persons AS p ON p.person_id = e.person_id
    WHERE e.model_version = %s
      AND p.status = 'active'
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
      AND p.status = 'active'
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
class Result:
    person_ids: tuple[str, ...]
    elapsed_ms: float


def _exact_query(conn: Any, vector: np.ndarray, model: str, top_k: int) -> Result:
    started = time.perf_counter_ns()
    with conn.cursor() as cur:
        cur.execute(EXACT_SQL, (vector, vector, model, top_k))
        rows = cur.fetchall()
    return Result(tuple(str(row[0]) for row in rows), (time.perf_counter_ns() - started) / 1_000_000)


def _hnsw_query(
    conn: Any,
    vector: np.ndarray,
    model: str,
    candidate_pool: int,
    top_k: int,
) -> Result:
    started = time.perf_counter_ns()
    with conn.cursor() as cur:
        cur.execute(HNSW_SQL, (vector, model, vector, candidate_pool, top_k))
        rows = cur.fetchall()
    return Result(tuple(str(row[0]) for row in rows), (time.perf_counter_ns() - started) / 1_000_000)


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(np.ceil((p / 100) * len(ordered))) - 1))
    return ordered[index]


def _load_queries(conn: Any, model: str, requested_ids: list[str] | None) -> list[tuple[str, np.ndarray]]:
    sql = """
    SELECT DISTINCT ON (person_id) person_id, embedding
    FROM face_embeddings
    WHERE model_version = %s
    ORDER BY person_id, quality_score DESC, id ASC;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (model,))
        rows = cur.fetchall()
    wanted = set(requested_ids or [])
    result = []
    for person_id, embedding in rows:
        pid = str(person_id)
        if wanted and pid not in wanted:
            continue
        vector = np.asarray(embedding, dtype=np.float32)
        norm = np.linalg.norm(vector)
        if norm <= 1e-6:
            continue
        result.append((pid, vector / norm))
    return result


def main() -> None:
    database_url = os.environ["DATABASE_URL"]
    model = os.getenv("MODEL_VERSION", "arcface_v1")
    top_k = int(os.getenv("TOP_K", "1"))
    candidate_pool = int(os.getenv("HNSW_CANDIDATE_POOL", "100"))
    ef_search = int(os.getenv("HNSW_EF_SEARCH", "100"))
    warmup = int(os.getenv("WARMUP", "20"))
    repeats = int(os.getenv("REPEATS", "100"))
    requested = [x for x in os.getenv("QUERY_PERSON_IDS", "").split(",") if x]

    if candidate_pool < top_k:
        raise ValueError("HNSW_CANDIDATE_POOL must be >= TOP_K")

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

        queries = _load_queries(conn, model, requested)
        if not queries:
            raise RuntimeError("No real benchmark queries found for MODEL_VERSION")

        exact_results = {pid: _exact_query(conn, vector, model, top_k) for pid, vector in queries}

        with conn.cursor() as cur:
            cur.execute("SET hnsw.ef_search = %s;", (ef_search,))

        for i in range(warmup):
            _, vector = queries[i % len(queries)]
            _hnsw_query(conn, vector, model, candidate_pool, top_k)

        hnsw_results: dict[str, Result] = {}
        all_latencies: list[float] = []
        for pid, vector in queries:
            samples = [_hnsw_query(conn, vector, model, candidate_pool, top_k) for _ in range(repeats)]
            all_latencies.extend(sample.elapsed_ms for sample in samples)
            hnsw_results[pid] = samples[-1]

        hits = sum(
            1
            for pid in exact_results
            if set(hnsw_results[pid].person_ids[:top_k]) == set(exact_results[pid].person_ids[:top_k])
        )
        recall_at_k = hits / len(exact_results)

        print({
            "model_version": model,
            "active_people": active_people,
            "embeddings": embeddings,
            "queries": len(queries),
            "top_k": top_k,
            "hnsw_candidate_pool": candidate_pool,
            "hnsw_ef_search": ef_search,
            "recall_at_k_vs_exact": recall_at_k,
            "hnsw_latency_ms_p50": _percentile(all_latencies, 50),
            "hnsw_latency_ms_p95": _percentile(all_latencies, 95),
            "hnsw_latency_ms_p99": _percentile(all_latencies, 99),
        })


if __name__ == "__main__":
    main()
