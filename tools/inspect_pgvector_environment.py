"""Inspect the real PostgreSQL/pgvector environment before retrieval benchmarking.

This is intentionally read-only. It never mutates application tables or settings.
It reports the extension version, PostgreSQL version, HNSW-related settings, and
indexes relevant to the face embedding retrieval benchmark.
"""

from __future__ import annotations

import json
import os
import sys

import psycopg


QUERIES = {
    "postgres_version": "SELECT version()",
    "pgvector_version": "SELECT extversion FROM pg_extension WHERE extname = 'vector'",
    "hnsw_ef_search": "SELECT current_setting('hnsw.ef_search', true)",
    "hnsw_iterative_scan": "SELECT current_setting('hnsw.iterative_scan', true)",
    "hnsw_max_scan_tuples": "SELECT current_setting('hnsw.max_scan_tuples', true)",
    "hnsw_scan_mem_multiplier": "SELECT current_setting('hnsw.scan_mem_multiplier', true)",
    "embedding_indexes": """
        SELECT schemaname, tablename, indexname, indexdef
        FROM pg_indexes
        WHERE tablename IN ('face_embeddings', 'benchmark_person_prototypes')
          AND indexdef ILIKE '%hnsw%'
        ORDER BY schemaname, tablename, indexname
    """,
}


def main() -> int:
    dsn = os.environ.get("PG_DSN")
    if not dsn:
        print(json.dumps({"status": "blocked", "reason": "PG_DSN is not set"}))
        return 2

    result: dict[str, object] = {"status": "ok"}
    try:
        with psycopg.connect(dsn, connect_timeout=5) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector')")
                result["pgvector_installed"] = bool(cur.fetchone()[0])
                if not result["pgvector_installed"]:
                    result["status"] = "blocked"
                    result["reason"] = "pgvector extension is not installed"
                    print(json.dumps(result, indent=2))
                    return 3

                for name, sql in QUERIES.items():
                    cur.execute(sql)
                    if name == "embedding_indexes":
                        result[name] = [
                            {
                                "schema": row[0],
                                "table": row[1],
                                "index": row[2],
                                "definition": row[3],
                            }
                            for row in cur.fetchall()
                        ]
                    else:
                        row = cur.fetchone()
                        result[name] = row[0] if row else None
    except Exception as exc:
        print(json.dumps({"status": "blocked", "reason": str(exc)}))
        return 4

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
