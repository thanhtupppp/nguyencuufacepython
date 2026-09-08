"""Synthetic pgvector retrieval benchmark with an exact person-level reference.

The benchmark intentionally uses deterministic synthetic unit embeddings so it can
run in CI without shipping biometric data. It measures retrieval correctness,
not face-recognition accuracy.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import psycopg


def make_gallery(persons: int, templates: int, dim: int, seed: int) -> list[tuple[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    rows: list[tuple[str, np.ndarray]] = []
    for i in range(persons):
        base = rng.normal(size=dim).astype(np.float32)
        base /= np.linalg.norm(base)
        for t in range(templates):
            noise = rng.normal(scale=0.03, size=dim).astype(np.float32)
            v = base + noise
            v /= np.linalg.norm(v)
            rows.append((f"bench-person-{i:05d}", v))
    return rows


def exact_person_topk(rows: list[tuple[str, np.ndarray]], q: np.ndarray, k: int) -> list[str]:
    best: dict[str, float] = {}
    for pid, v in rows:
        score = float(np.dot(v, q))
        best[pid] = max(best.get(pid, -1.0), score)
    return [pid for pid, _ in sorted(best.items(), key=lambda x: x[1], reverse=True)[:k]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--persons", type=int, default=1000)
    ap.add_argument("--templates", type=int, default=10)
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--queries", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20260908)
    args = ap.parse_args()

    dsn = os.environ.get("PG_DSN", "postgresql://face_admin:ci@127.0.0.1:5432/face_recognition")
    rng = np.random.default_rng(args.seed)
    rows = make_gallery(args.persons, args.templates, args.dim, args.seed)

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE face_embeddings, persons RESTART IDENTITY CASCADE")
            cur.executemany(
                "INSERT INTO persons(person_id, name, status) VALUES (%s, %s, 'active')",
                [(pid, pid) for pid in sorted({p for p, _ in rows})],
            )
            cur.executemany(
                "INSERT INTO face_embeddings(person_id, embedding, model_version) VALUES (%s, %s, 'arcface_v1')",
                [(pid, v.tolist()) for pid, v in rows],
            )
        conn.commit()

        sql = """
        SELECT person_id, 1.0 - (embedding <=> %s::vector) AS similarity
        FROM face_embeddings
        WHERE model_version = 'arcface_v1'
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """
        recalls: list[float] = []
        latencies_ms: list[float] = []
        for _ in range(args.queries):
            pid, base = rows[int(rng.integers(0, len(rows)))]
            q = base.copy()
            q += rng.normal(scale=0.01, size=args.dim).astype(np.float32)
            q /= np.linalg.norm(q)
            expected = set(exact_person_topk(rows, q, args.top_k))
            t0 = time.perf_counter()
            with conn.cursor() as cur:
                cur.execute(sql, (q.tolist(), q.tolist(), args.top_k * args.templates))
                raw = cur.fetchall()
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)
            got: list[str] = []
            for p, _ in raw:
                if p not in got:
                    got.append(p)
                if len(got) == args.top_k:
                    break
            recalls.append(len(expected.intersection(got)) / args.top_k)

    print({
        "persons": args.persons,
        "templates_per_person": args.templates,
        "queries": args.queries,
        "top_k": args.top_k,
        "recall_at_k_mean": float(np.mean(recalls)),
        "recall_at_k_min": float(np.min(recalls)),
        "latency_ms_p50": float(np.percentile(latencies_ms, 50)),
        "latency_ms_p95": float(np.percentile(latencies_ms, 95)),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
