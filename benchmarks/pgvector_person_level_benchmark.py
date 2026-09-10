"""Benchmark person-level exact vs HNSW pgvector retrieval.

This harness intentionally does not generate synthetic embeddings or claim
accuracy. It requires a real PostgreSQL/pgvector database populated with a
frozen benchmark dataset and compares exact cosine retrieval with HNSW under
identical queries.

Environment:
  DATABASE_URL=postgresql://...
  MODEL_VERSION=arcface_v1
  TOP_K=1
  HNSW_EF_SEARCH=100
  WARMUP=20
  REPEATS=100
  QUERY_PERSON_IDS=person_001,person_002,... (optional)

The benchmark reports recall@K against exact person-level results and
latency percentiles. Threshold decisions are deliberately outside this
benchmark.
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


@dataclass(frozen=True)
class Result:
    person_ids: tuple[str, ...]
    elapsed_ms: float


def _query(conn: Any, sql: str, vector: np.ndarray, model: str, top_k: int) -> Result:
    started = time.perf_counter_ns()
    with conn.cursor() as cur:
        cur.execute(sql, (vector, vector, model, top_k))
        rows = cur.fetchall()
    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
    return Result(tuple(str(row[0]) for row in rows), elapsed_ms)


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
    ef_search = int(os.getenv("HNSW_EF_SEARCH", "100"))
    warmup = int(os.getenv("WARMUP", "20"))
    repeats = int(os.getenv("REPEATS", "100"))
    requested = [x for x in os.getenv("QUERY_PERSON_IDS", "").split(",") if x]

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

        # Exact search is the ground truth for this benchmark.
        exact_results: dict[str, Result] = {}
        for pid, vector in queries:
            exact_results[pid] = _query(conn, EXACT_SQL, vector, model, top_k)

        with conn.cursor() as cur:
            cur.execute("SET hnsw.ef_search = %s;", (ef_search,))

        for _ in range(warmup):
            pid, vector = queries[_ % len(queries)]
            _query(conn, HNSW_SQL, vector, model, top_k)

        hnsw_results: dict[str, Result] = {}
        for pid, vector in queries:
            samples = [_query(conn, HNSW_SQL, vector, model, top_k) for _ in range(repeats)]
            hnsw_results[pid] = Result(samples[-1].person_ids, statistics.median(s.elapsed_ms for s in samples))

        hits = sum(
            1 for pid in exact_results
            if hnsw_results[pid].person_ids[:top_k] == exact_results[pid].person_ids[:top_k]
        )
        recall_at_k = hits / len(exact_results)
        latencies = [hnsw_results[pid].elapsed_ms for pid in hnsw_results]

        print({
            "model_version": model,
            "active_people": active_people,
            "embeddings": embeddings,
            "queries": len(queries),
            "top_k": top_k,
            "hnsw_ef_search": ef_search,
            "recall_at_k_vs_exact": recall_at_k,
            "hnsw_latency_ms_p50": _percentile(latencies, 50),
            "hnsw_latency_ms_p95": _percentile(latencies, 95),
            "hnsw_latency_ms_p99": _percentile(latencies, 99),
        })


if __name__ == "__main__":
    main()
