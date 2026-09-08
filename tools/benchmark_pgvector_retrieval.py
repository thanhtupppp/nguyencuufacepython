"""Deterministic pgvector person-level retrieval benchmark.

The benchmark compares correctness-first production retrieval with
candidate-first HNSW and one-vector-per-person prototype retrieval.
Synthetic vectors are used so no biometric data is required.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import psycopg


PRODUCTION_SQL = """
SELECT person_id, similarity, name
FROM (
    SELECT e.person_id,
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

CANDIDATE_SQL = """
SELECT e.person_id, 1.0 - (e.embedding <=> %s::vector) AS similarity
FROM face_embeddings AS e
JOIN persons AS p ON p.person_id = e.person_id
WHERE e.model_version = %s
  AND (%s = FALSE OR p.status = 'active')
ORDER BY e.embedding <=> %s::vector
LIMIT %s;
"""

PROTOTYPE_SQL = """
SELECT person_id, 1.0 - (embedding <=> %s::vector) AS similarity
FROM benchmark_person_prototypes
WHERE active = TRUE
ORDER BY embedding <=> %s::vector
LIMIT %s;
"""


def make_gallery(persons: int, templates: int, dim: int, seed: int) -> list[tuple[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    rows: list[tuple[str, np.ndarray]] = []
    for i in range(persons):
        base = rng.normal(size=dim).astype(np.float32)
        base /= np.linalg.norm(base)
        for _ in range(templates):
            noise = rng.normal(scale=0.03, size=dim).astype(np.float32)
            v = base + noise
            v /= np.linalg.norm(v)
            rows.append((f"bench-person-{i:05d}", v))
    return rows


def exact_person_topk(rows, q, k, active, model_version):
    if model_version != "arcface_v1":
        return []
    best = {}
    for pid, v in rows:
        if pid not in active:
            continue
        score = float(np.dot(v, q))
        best[pid] = max(best.get(pid, -1.0), score)
    return [pid for pid, _ in sorted(best.items(), key=lambda x: x[1], reverse=True)[:k]]


def unique_topk(raw_rows, k):
    out = []
    for pid, _ in raw_rows:
        if pid not in out:
            out.append(pid)
        if len(out) == k:
            break
    return out


def exact_rerank_candidates(raw_rows, q, gallery, k):
    candidate_ids = {pid for pid, _ in raw_rows}
    scored = []
    for pid in candidate_ids:
        score = max(float(np.dot(v, q)) for v in gallery[pid])
        scored.append((pid, score))
    scored.sort(key=lambda item: item[1], reverse=True)
    return [pid for pid, _ in scored[:k]]


def build_prototypes(gallery):
    prototypes = {}
    for pid, vectors in gallery.items():
        proto = np.mean(np.stack(vectors), axis=0)
        norm = np.linalg.norm(proto)
        if norm == 0:
            raise ValueError(f"zero prototype for {pid}")
        prototypes[pid] = (proto / norm).astype(np.float32)
    return prototypes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--persons", type=int, default=1000)
    ap.add_argument("--templates", type=int, default=10)
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--queries", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20260908)
    ap.add_argument("--active-ratio", type=float, default=1.0)
    ap.add_argument("--ef-search", type=int, default=100)
    ap.add_argument("--iterative-scan", choices=("off", "strict_order", "relaxed_order"), default="off")
    ap.add_argument(
        "--mode",
        choices=("production", "candidate", "candidate_rerank", "prototype", "prototype_rerank"),
        default="production",
    )
    ap.add_argument("--oversample", type=int, default=20)
    args = ap.parse_args()

    if not 0.0 < args.active_ratio <= 1.0:
        raise SystemExit("--active-ratio must be in (0, 1]")
    if args.oversample < 1:
        raise SystemExit("--oversample must be >= 1")
    if args.ef_search < 1:
        raise SystemExit("--ef-search must be >= 1")

    dsn = os.environ.get("PG_DSN", "postgresql://face_admin:ci@127.0.0.1:5432/face_recognition")
    rng = np.random.default_rng(args.seed)
    rows = make_gallery(args.persons, args.templates, args.dim, args.seed)
    person_ids = sorted({p for p, _ in rows})
    active_count = max(1, int(round(len(person_ids) * args.active_ratio)))
    active = set(person_ids[:active_count])
    gallery = {}
    for pid, vector in rows:
        gallery.setdefault(pid, []).append(vector)
    prototypes = build_prototypes(gallery)

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE face_embeddings, persons RESTART IDENTITY CASCADE")
            cur.executemany(
                "INSERT INTO persons(person_id, name, status) VALUES (%s, %s, %s)",
                [(pid, pid, "active" if pid in active else "inactive") for pid in person_ids],
            )
            cur.executemany(
                "INSERT INTO face_embeddings(person_id, embedding, model_version) VALUES (%s, %s, 'arcface_v1')",
                [(pid, v.tolist()) for pid, v in rows],
            )
            cur.execute("DROP TABLE IF EXISTS benchmark_person_prototypes")
            cur.execute(
                "CREATE TEMP TABLE benchmark_person_prototypes (person_id text PRIMARY KEY, embedding vector(512), active boolean)"
            )
            cur.executemany(
                "INSERT INTO benchmark_person_prototypes(person_id, embedding, active) VALUES (%s, %s, %s)",
                [(pid, v.tolist(), pid in active) for pid, v in prototypes.items()],
            )
            cur.execute(
                "CREATE INDEX benchmark_person_prototypes_hnsw ON benchmark_person_prototypes USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)"
            )
        conn.commit()

        with conn.cursor() as cur:
            cur.execute("SELECT set_config('hnsw.ef_search', %s, false)", (str(args.ef_search),))
            cur.execute("SELECT set_config('hnsw.iterative_scan', %s, false)", (args.iterative_scan,))

        recalls = []
        latencies_ms = []
        for _ in range(args.queries):
            _, base = rows[int(rng.integers(0, len(rows)))]
            q = base.copy()
            q += rng.normal(scale=0.01, size=args.dim).astype(np.float32)
            q /= np.linalg.norm(q)
            expected = set(exact_person_topk(rows, q, args.top_k, active, "arcface_v1"))
            t0 = time.perf_counter()
            with conn.cursor() as cur:
                if args.mode == "production":
                    cur.execute(PRODUCTION_SQL, (q.tolist(), q.tolist(), "arcface_v1", True, args.top_k))
                elif args.mode in ("candidate", "candidate_rerank"):
                    candidate_limit = args.top_k * args.oversample
                    cur.execute(CANDIDATE_SQL, (q.tolist(), "arcface_v1", True, q.tolist(), candidate_limit))
                else:
                    candidate_limit = args.top_k * args.oversample
                    cur.execute(PROTOTYPE_SQL, (q.tolist(), q.tolist(), candidate_limit))
                raw = cur.fetchall()
            if args.mode == "production":
                got = [row[0] for row in raw]
            elif args.mode == "candidate_rerank":
                got = exact_rerank_candidates(raw, q, gallery, args.top_k)
            elif args.mode == "prototype_rerank":
                got = exact_rerank_candidates(raw, q, gallery, args.top_k)
            else:
                got = unique_topk(raw, args.top_k)
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)
            recalls.append(len(expected.intersection(got)) / args.top_k)

    print({
        "mode": args.mode,
        "persons": args.persons,
        "templates_per_person": args.templates,
        "active_ratio": args.active_ratio,
        "queries": args.queries,
        "top_k": args.top_k,
        "ef_search": args.ef_search,
        "iterative_scan": args.iterative_scan,
        "oversample": args.oversample if args.mode != "production" else None,
        "recall_at_k_mean": float(np.mean(recalls)),
        "recall_at_k_min": float(np.min(recalls)),
        "missing_person_rate": float(1.0 - np.mean(recalls)),
        "latency_ms_p50": float(np.percentile(latencies_ms, 50)),
        "latency_ms_p95": float(np.percentile(latencies_ms, 95)),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
