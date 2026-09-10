"""Benchmark pgvector person-level retrieval variants against exact reference.

Requires DATABASE_URL and a PostgreSQL instance with pgvector. This tool is
measurement-only: it never changes recognition thresholds and never writes to
production tables.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from dataclasses import dataclass

import numpy as np
import psycopg


@dataclass(frozen=True)
class Scenario:
    people: int
    templates: tuple[int, ...]
    inactive_fraction: float = 0.0
    model_fraction: float = 1.0


def unit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / max(float(np.linalg.norm(x)), 1e-12)


def gallery(s: Scenario, seed: int = 20260910):
    rng = np.random.default_rng(seed + s.people + sum(s.templates))
    rows, queries = [], []
    for person in range(s.people):
        center = unit(rng.normal(size=512).astype(np.float32))
        status = "inactive" if (person / max(s.people, 1)) < s.inactive_fraction else "active"
        model = "arcface_v1" if (person / max(s.people, 1)) < s.model_fraction else "arcface_other"
        count = s.templates[person % len(s.templates)]
        for t in range(count):
            emb = unit(center + rng.normal(0, 0.035, 512).astype(np.float32))
            rows.append((f"p{person:05d}", t, model, status, emb))
        queries.append(unit(center + rng.normal(0, 0.025, 512).astype(np.float32)))
    return rows, queries


def exact(rows, q, k):
    best = {}
    for person, _t, model, status, emb in rows:
        if model != "arcface_v1" or status != "active":
            continue
        best[person] = max(best.get(person, -1.0), float(np.dot(emb, q)))
    return [p for p, _ in sorted(best.items(), key=lambda x: x[1], reverse=True)[:k]]


def pct(xs, q):
    return float(np.percentile(np.asarray(xs), q)) if xs else 0.0


def run(cur, sql, queries, expected, k, params):
    lat, recalls, missing, dup = [], [], 0, 0
    for i, q in enumerate(queries):
        t0 = time.perf_counter_ns()
        cur.execute(sql, params(q, k))
        got = [str(r[0]) for r in cur.fetchall()]
        lat.append((time.perf_counter_ns() - t0) / 1e6)
        missing += len(got) < k
        dup += len(got) != len(set(got))
        recalls.append(len(set(got) & set(expected[i])) / k)
    return {
        "recall": statistics.mean(recalls),
        "missing_result_rate": missing / len(queries),
        "duplicate_person_result_rate": dup / len(queries),
        "latency_ms": {"p50": pct(lat, 50), "p95": pct(lat, 95), "p99": pct(lat, 99)},
    }


PERSON_SQL = """
SELECT person_id, similarity FROM (
 SELECT e.person_id, 1.0 - (e.embedding <=> %s::vector) AS similarity,
 ROW_NUMBER() OVER (PARTITION BY e.person_id ORDER BY e.embedding <=> %s::vector ASC) AS rn
 FROM benchmark_face_embeddings e JOIN benchmark_persons p ON p.person_id=e.person_id
 WHERE e.model_version=%s AND p.status='active'
) ranked WHERE rn=1 ORDER BY similarity DESC LIMIT %s;
"""

CANDIDATE_SQL = """
WITH candidates AS (
 SELECT e.person_id, e.embedding <=> %s::vector AS distance
 FROM benchmark_face_embeddings e JOIN benchmark_persons p ON p.person_id=e.person_id
 WHERE e.model_version=%s AND p.status='active'
 ORDER BY e.embedding <=> %s::vector LIMIT %s
)
SELECT person_id, 1.0-min(distance) AS similarity FROM candidates
GROUP BY person_id ORDER BY min(distance) LIMIT %s;
"""


def explain(cur, sql, params):
    cur.execute("EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) " + sql, params)
    return cur.fetchone()[0][0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="reports/vector_ann.json")
    ap.add_argument("--repeats", type=int, default=20)
    args = ap.parse_args()
    url = os.getenv("DATABASE_URL")
    if not url:
        raise SystemExit("DATABASE_URL is required")

    scenarios = [
        Scenario(100, (1,)), Scenario(100, (10,)), Scenario(1000, (10,)),
        Scenario(10000, (10,)), Scenario(100, (1, 3, 10, 30)),
        Scenario(100, (10,), .5), Scenario(100, (10,), .9),
        Scenario(1000, (10,), 0.0, .5),
    ]
    report = {"protocol_version": "P1.1.2", "scenarios": []}
    with psycopg.connect(url) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute("SELECT version(), extversion FROM pg_extension WHERE extname='vector'")
            report["environment"] = dict(zip(("postgres_version", "pgvector_version"), cur.fetchone()))
            for s in scenarios:
                rows, queries = gallery(s)
                cur.execute("DROP TABLE IF EXISTS benchmark_face_embeddings")
                cur.execute("DROP TABLE IF EXISTS benchmark_persons")
                cur.execute("CREATE TABLE benchmark_persons(person_id text PRIMARY KEY,status text NOT NULL)")
                cur.execute("CREATE TABLE benchmark_face_embeddings(id bigserial PRIMARY KEY,person_id text NOT NULL,embedding vector(512) NOT NULL,model_version text NOT NULL)")
                cur.executemany("INSERT INTO benchmark_persons VALUES(%s,%s)", sorted({(r[0], r[3]) for r in rows}))
                cur.executemany("INSERT INTO benchmark_face_embeddings(person_id,embedding,model_version) VALUES(%s,%s,%s)", [(r[0], r[4].tolist(), r[2]) for r in rows])
                cur.execute("CREATE INDEX benchmark_hnsw ON benchmark_face_embeddings USING hnsw (embedding vector_cosine_ops)")
                expected = {k: [exact(rows, q, k) for q in queries] for k in (1, 5, 10)}
                for ef in (40, 80, 160, 320):
                    cur.execute("SET LOCAL hnsw.ef_search = %s", (ef,))
                    for iterative in ("off", "strict_order", "relaxed_order"):
                        try:
                            cur.execute("SET LOCAL hnsw.iterative_scan = %s", (iterative,))
                        except psycopg.Error:
                            conn.rollback(); conn.autocommit = True; cur = conn.cursor()
                            if iterative != "off":
                                continue
                        for k in (1, 5, 10):
                            def params(q, kk, k=k):
                                return (q.tolist(), q.tolist(), "arcface_v1", kk)
                            ms = run(cur, PERSON_SQL, queries, expected[k], k, params)
                            report["scenarios"].append({"people": s.people, "templates": s.templates, "inactive_fraction": s.inactive_fraction, "model_fraction": s.model_fraction, "variant": "person_level", "ef_search": ef, "iterative_scan": iterative, "k": k, "measurements": ms})
                for k in (1, 5, 10):
                    def cparams(q, kk, k=k):
                        return (q.tolist(), "arcface_v1", q.tolist(), max(kk * 10, 100), kk)
                    ms = run(cur, CANDIDATE_SQL, queries, expected[k], k, cparams)
                    report["scenarios"].append({"people": s.people, "templates": s.templates, "inactive_fraction": s.inactive_fraction, "model_fraction": s.model_fraction, "variant": "candidate_first", "k": k, "measurements": ms})
                try:
                    cur.execute("SET hnsw.iterative_scan = 'off'")
                    params = (queries[0].tolist(), queries[0].tolist(), "arcface_v1", 5)
                    report.setdefault("explain", []).append({"scenario": s.people, "variant": "person_level", "plan": explain(cur, PERSON_SQL, params)})
                except psycopg.Error:
                    pass
                cur.execute("DROP TABLE benchmark_face_embeddings")
                cur.execute("DROP TABLE benchmark_persons")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
