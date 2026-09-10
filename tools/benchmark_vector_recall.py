"""Benchmark person-level pgvector recall/latency against exact reference."""
from __future__ import annotations
import argparse, json, os, statistics, time
from dataclasses import dataclass
import numpy as np
try:
    import psycopg
except ImportError as exc:
    raise SystemExit("psycopg is required") from exc

@dataclass(frozen=True)
class Scenario:
    people: int
    templates: tuple[int, ...]
    inactive_fraction: float = 0.0

def normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / max(float(np.linalg.norm(x)), 1e-12)

def make_gallery(s: Scenario, seed: int = 20260910):
    rng = np.random.default_rng(seed + s.people + sum(s.templates))
    rows, queries = [], []
    for person in range(s.people):
        center = normalize(rng.normal(size=512).astype(np.float32))
        status = "inactive" if (person / max(s.people, 1)) < s.inactive_fraction else "active"
        count = s.templates[person % len(s.templates)]
        for t in range(count):
            emb = normalize(center + rng.normal(0, 0.035, 512).astype(np.float32))
            rows.append((f"p{person:05d}", t, "arcface_v1", status, emb))
        queries.append(normalize(center + rng.normal(0, 0.025, 512).astype(np.float32)))
    return rows, queries

def exact_topk(rows, query, k):
    best = {}
    for person, _t, model, status, emb in rows:
        if model != "arcface_v1" or status != "active":
            continue
        best[person] = max(float(np.dot(emb, query)), best.get(person, -1.0))
    return [p for p, _ in sorted(best.items(), key=lambda x: x[1], reverse=True)[:k]]

def pct(values, q):
    return float(np.percentile(np.asarray(values), q)) if values else 0.0

SQL = """
SELECT person_id, similarity FROM (
 SELECT e.person_id, 1.0 - (e.embedding <=> %s::vector) AS similarity,
 ROW_NUMBER() OVER (PARTITION BY e.person_id ORDER BY e.embedding <=> %s::vector ASC) AS rn
 FROM benchmark_face_embeddings e JOIN benchmark_persons p ON p.person_id=e.person_id
 WHERE e.model_version=%s AND p.status='active'
) ranked WHERE rn=1 ORDER BY similarity DESC LIMIT %s;
"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="reports/vector_recall.json")
    ap.add_argument("--repeats", type=int, default=20)
    args = ap.parse_args()
    if not os.getenv("DATABASE_URL"):
        raise SystemExit("DATABASE_URL is required; no synthetic benchmark is reported as project evidence")
    scenarios = [Scenario(100,(1,)), Scenario(100,(10,)), Scenario(1000,(10,)), Scenario(10000,(10,)), Scenario(100,(1,3,10,30)), Scenario(100,(10,),.5), Scenario(100,(10,),.9)]
    report={"protocol_version":"P1.1.1","scenarios":[]}
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        conn.autocommit=True
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute("DROP TABLE IF EXISTS benchmark_face_embeddings")
            cur.execute("DROP TABLE IF EXISTS benchmark_persons")
            cur.execute("CREATE TABLE benchmark_persons(person_id text PRIMARY KEY,status text NOT NULL)")
            cur.execute("CREATE TABLE benchmark_face_embeddings(id bigserial PRIMARY KEY,person_id text NOT NULL,embedding vector(512) NOT NULL,model_version text NOT NULL)")
            cur.execute("SELECT version(), extversion FROM pg_extension WHERE extname='vector'")
            pv,vv=cur.fetchone(); report["environment"]={"postgres_version":pv,"pgvector_version":vv}
            for s in scenarios:
                rows,queries=make_gallery(s); cur.execute("TRUNCATE benchmark_face_embeddings,benchmark_persons")
                cur.executemany("INSERT INTO benchmark_persons VALUES(%s,%s)", sorted({(r[0],r[3]) for r in rows}))
                cur.executemany("INSERT INTO benchmark_face_embeddings(person_id,embedding,model_version) VALUES(%s,%s,%s)",[(r[0],r[4].tolist(),r[2]) for r in rows])
                ms={}
                for k in (1,5,10):
                    expected=[exact_topk(rows,q,k) for q in queries]; lat=[]; rec=[]; miss=dup=0
                    for i,q in enumerate(queries):
                        t=time.perf_counter_ns(); cur.execute(SQL,(q.tolist(),q.tolist(),"arcface_v1",k)); got=[str(r[0]) for r in cur.fetchall()]; lat.append((time.perf_counter_ns()-t)/1e6)
                        miss += len(got)<k; dup += len(got)!=len(set(got)); rec.append(len(set(got)&set(expected[i]))/k)
                    ms[str(k)]={"recall":statistics.mean(rec),"missing_result_rate":miss/len(queries),"duplicate_person_result_rate":dup/len(queries),"latency_ms":{"p50":pct(lat,50),"p95":pct(lat,95),"p99":pct(lat,99)}}
                report["scenarios"].append({"people":s.people,"templates":s.templates,"inactive_fraction":s.inactive_fraction,"measurements":ms})
            cur.execute("DROP TABLE benchmark_face_embeddings"); cur.execute("DROP TABLE benchmark_persons")
    os.makedirs(os.path.dirname(args.output) or ".",exist_ok=True)
    with open(args.output,"w",encoding="utf-8") as f: json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))

if __name__ == "__main__": main()
