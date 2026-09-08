"""Deterministic pgvector person-level retrieval benchmark.

The benchmark compares the correctness-first production query with a
candidate-first HNSW strategy against the same exact person-level reference.
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

CANDIDATE_SQL = """
SELECT e.person_id, 1.0 - (e.embedding <=> %s::vector) AS similarity
FROM face_embeddings AS e
JOIN persons AS p ON p.person_id = e.person_id
WHERE e.model_version = %s
  AND (%s = FALSE OR p.status = 'active')
ORDER BY e.embedding <=> %s::vector
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


def exact_person_topk(
    rows: list[tuple[str, np.ndarray]],
    q: np.ndarray,
    k: int,
    active: set[str],
    model_version: str,
) -> list[str]:
    # Synthetic rows all carry the requested model version. The argument is
    # kept explicit so the reference semantics mirror production filtering.
    if model_version != "arcface_v1":
        return []
    best: dict[str, float] = {}
    for pid, v in rows:
        if pid not in active:
            continue
        score = float(np.dot(v, q))
        best[pid] = max(best.get(pid, -1.0), score)
    return [pid for pid, _ in sorted(best.items(), key=lambda x: x[1], reverse=True)[:k]]


def unique_topk(raw_rows: list[tuple[str, float]], k: int) -> list[str]:
    out: list[str] = []
    for pid, _ in raw_rows:
        if pid not in out:
            out.append(pid)
        if len(out) == k:
            break
    return out


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
    ap.add_argument(
        "--iterative-scan",
        choices=("off", "strict_order", "relaxed_order"),
        default="off",
    )
    ap.add_argument(
        "--mode",
        choices=("production", "candidate"),
        default="production",
        help="production=best-template-per-person query; candidate=HNSW template oversampling",
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
        conn.commit()

        with conn.cursor() as cur:
            # PostgreSQL does not accept bind parameters in SET's GUC value.
            # set_config() keeps the benchmark parameterized without unsafe SQL interpolation.
            cur.execute("SELECT set_config('hnsw.ef_search', %s, false)", (str(args.ef_search),))
            cur.execute(
                "SELECT set_config('hnsw.iterative_scan', %s, false)",
                (args.iterative_scan,),
            )

        recalls: list[float] = []
        latencies_ms: list[float] = []
        for _ in range(args.queries):
            _, base = rows[int(rng.integers(0, len(rows)))]
            q = base.copy()
            q += rng.normal(scale=0.01, size=args.dim).astype(np.float32)
            q /= np.linalg.norm(q)
            expected = set(exact_person_topk(rows, q, args.top_k, active, "arcface_v1"))

            t0 = time.perf_counter()
            with conn.cursor() as cur:
                if args.mode == "production":
                    cur.execute(
                        PRODUCTION_SQL,
                        (q.tolist(), q.tolist(), "arcface_v1", True, args.top_k),
                    )
                else:
                    candidate_limit = args.top_k * args.oversample
                    cur.execute(
                        CANDIDATE_SQL,
                        (
                            q.tolist(),
                            "arcface_v1",
                            True,
                            q.tolist(),
                            candidate_limit,
                        ),
                    )
                raw = cur.fetchall()
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)
            got = [p for p in (row[0] for row in raw)] if args.mode == "production" else unique_topk(raw, args.top_k)
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
        "oversample": args.oversample if args.mode == "candidate" else None,
        "recall_at_k_mean": float(np.mean(recalls)),
        "recall_at_k_min": float(np.min(recalls)),
        "latency_ms_p50": float(np.percentile(latencies_ms, 50)),
        "latency_ms_p95": float(np.percentile(latencies_ms, 95)),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
