"""Runtime readiness probe for the real PostgreSQL + pgvector backend."""
from __future__ import annotations

from typing import Any


REQUIRED_TABLES = ("persons", "face_embeddings")


def probe_postgres_pgvector(db: Any) -> tuple[bool, str]:
    """Return (ready, detail) without accepting memory/SQLite fallback as ready.

    Recognition production readiness must prove that the actual PostgreSQL
    connection is alive, pgvector is installed, and the core identity/vector
    tables exist. A local fallback is intentionally reported as not ready.
    """
    if db is None:
        return False, "database_client_missing"
    if getattr(db, "use_memory", False):
        return False, "memory_fallback_active"
    if getattr(db, "use_sqlite", False):
        return False, "sqlite_backend_active"

    conn = getattr(db, "_conn", None)
    if conn is None:
        return False, "postgres_connection_missing"

    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1;")
            if cur.fetchone() != (1,):
                return False, "basic_query_failed"

            cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector';")
            extension = cur.fetchone()
            if not extension:
                return False, "pgvector_extension_missing"

            cur.execute(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                  AND table_name = ANY(%s);
                """,
                [list(REQUIRED_TABLES)],
            )
            tables = {row[0] for row in cur.fetchall()}
            missing = [name for name in REQUIRED_TABLES if name not in tables]
            if missing:
                return False, "missing_tables:" + ",".join(missing)

        return True, f"postgres_pgvector_verified:{extension[0]}"
    except Exception as exc:  # pragma: no cover - exact driver exception varies
        return False, f"postgres_probe_error:{type(exc).__name__}"
