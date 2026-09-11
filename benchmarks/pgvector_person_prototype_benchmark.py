"""Leakage-safe benchmark: template-level person grouping vs one-vector/person prototype.

This is a research benchmark only. It never changes production search.

Required:
  DATABASE_URL=postgresql://...
  MODEL_VERSION=arcface_v1

Optional:
  TOP_K=1
  REPEATS=20
  MIN_TEMPLATES_PER_PERSON=2
  ACTIVE_ONLY=1
  MAX_QUERIES=0

The query embedding is excluded from both the template gallery and the
prototype construction for its own person.
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


@dataclass(frozen=True)
class Query:
    embedding_id: int
    person_id: str
    vector: np.ndarray


def normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm <= 1e-8:
        raise ValueError("zero-norm embedding")
    return v.astype(np.float32) / norm


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
        WHERE e2.person_id = e.person_id AND e2.model_version = e.model_version
      ) >= %s
    ORDER BY e.person_id, e.id;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (model, active_only, min_templates))
        rows = cur.fetchall()
    result: list[Query] = []
    for embedding_id, person_id, embedding in rows:
        result.append(Query(int(embedding_id), str(person_id), normalize(np.asarray(embedding, dtype=np.float32))))
        if max_queries and len(result) >= max_queries:
            break
    return result


def template_exact(conn: Any, q: Query, model: str, top_k: int, active_only: bool) -> tuple[str, ...]:
    sql = """
    WITH ranked AS (
      SELECT e.person_id,
             e.embedding <=> %s::vector AS distance,
             ROW_NUMBER() OVER (
               PARTITION BY e.person_id
               ORDER BY e.embedding <=> %s::vector ASC, e.id ASC
             ) AS rn
      FROM face_embeddings e
      JOIN persons p ON p.person_id = e.person_id
      WHERE e.model_version = %s
        AND e.id <> %s
        AND (%s = FALSE OR p.status = 'active')
    )
    SELECT person_id
    FROM ranked
    WHERE rn = 1
    ORDER BY distance ASC, person_id ASC
    LIMIT %s;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (q.vector, q.vector, model, q.embedding_id, active_only, top_k))
        return tuple(str(row[0]) for row in cur.fetchall())


def build_prototype(conn: Any, q: Query, model: str, active_only: bool) -> tuple[np.ndarray, ...]:
    """Return all person prototypes, excluding q.embedding_id from its own person."""
    sql = """
    SELECT e.person_id, e.id, e.embedding, e.quality_score
    FROM face_embeddings e
    JOIN persons p ON p.person_id = e.person_id
    WHERE e.model_version = %s
      AND e.id <> %s
      AND (%s = FALSE OR p.status = 'active')
    ORDER BY e.person_id, e.id;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (model, q.embedding_id, active_only))
        rows = cur.fetchall()

    grouped: dict[str, list[tuple[np.ndarray, float]]] = {}
    for person_id, _embedding_id, embedding, quality_score in rows:
        vec = normalize(np.asarray(embedding, dtype=np.float32))
        weight = max(float(quality_score or 1.0), 1e-3)
        grouped.setdefault(str(person_id), []).append((vec, weight))

    prototypes: list[tuple[str, np.ndarray]] = []
    for person_id, items in grouped.items():
        total = sum(weight for _, weight in items)
        centroid = sum(vec * weight for vec, weight in items) / total
        prototypes.append((person_id, normalize(centroid)))
    return tuple(prototypes)


def prototype_rank(prototypes: tuple[tuple[str, np.ndarray], ...], q: Query, top_k: int) -> tuple[str, ...]:
    ranked = sorted(
        ((person_id, float(np.dot(proto, q.vector))) for person_id, proto in prototypes),
        key=lambda item: (-item[1], item[0]),
    )
    return tuple(person_id for person_id, _ in ranked[:top_k])


def main() -> None:
    database_url = os.environ["DATABASE_URL"]
    model = os.getenv("MODEL_VERSION", "arcface_v1")
    top_k = int(os.getenv("TOP_K", "1"))
    repeats = int(os.getenv("REPEATS", "20"))
    min_templates = int(os.getenv("MIN_TEMPLATES_PER_PERSON", "2"))
    active_only = os.getenv("ACTIVE_ONLY", "1") not in {"0", "false", "False"}
    max_queries = int(os.getenv("MAX_QUERIES", "0"))

    if top_k < 1:
        raise ValueError("TOP_K must be >= 1")

    with psycopg.connect(database_url) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector';")
            extension = cur.fetchone()
            if not extension:
                raise RuntimeError("pgvector extension is not installed")

        queries = load_queries(conn, model, min_templates, active_only, max_queries)
        if not queries:
            raise RuntimeError("No leakage-safe queries found")

        template_recall = 0
        prototype_recall = 0
        prototype_agreement = 0
        latencies: list[float] = []

        for q in queries:
            reference = template_exact(conn, q, model, top_k, active_only)
            prototypes = build_prototype(conn, q, model, active_only)
            last = ()
            for _ in range(repeats):
                started = time.perf_counter_ns()
                last = prototype_rank(prototypes, q, top_k)
                latencies.append((time.perf_counter_ns() - started) / 1_000_000)
            if q.person_id in reference:
                template_recall += 1
            if q.person_id in last:
                prototype_recall += 1
            if last == reference:
                prototype_agreement += 1

        print({
            "benchmark": "template_vs_person_prototype",
            "model_version": model,
            "pgvector_version": extension[0],
            "queries": len(queries),
            "top_k": top_k,
            "template_reference_recall_at_k": template_recall / len(queries),
            "prototype_query_person_recall_at_k": prototype_recall / len(queries),
            "prototype_vs_template_agreement": prototype_agreement / len(queries),
            "prototype_rank_latency_ms_p50": percentile(latencies, 50),
            "prototype_rank_latency_ms_p95": percentile(latencies, 95),
            "prototype_rank_latency_ms_p99": percentile(latencies, 99),
            "prototype_rank_latency_ms_mean": statistics.mean(latencies) if latencies else 0.0,
        })


if __name__ == "__main__":
    main()
