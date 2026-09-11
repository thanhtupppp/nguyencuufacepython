"""Regression test for person-level top-k semantics in PostgreSQL/pgvector.

A person may have multiple enrollment embeddings. The database search must rank
individual people, not raw embedding rows, before applying LIMIT. Otherwise one
person with many embeddings can consume the LIMIT and hide the true second
candidate, corrupting the top1-top2 margin used by recognition.
"""

import os

import numpy as np
import pytest

try:
    import psycopg  # type: ignore
except ImportError:  # pragma: no cover
    psycopg = None

from src.database.client import DatabaseClient


pytestmark = pytest.mark.integration


def _database_client_from_env() -> DatabaseClient:
    return DatabaseClient(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        dbname=os.getenv("POSTGRES_DB", "face_recognition"),
        user=os.getenv("POSTGRES_USER", "face_admin"),
        password=os.getenv("POSTGRES_PASSWORD", "face_secure_password_2026"),
        fallback_to_memory=False,
    )


def test_postgres_search_returns_top_k_people_not_embedding_rows():
    """One person's many templates must not hide the second-best person."""
    if psycopg is None:
        pytest.fail("psycopg is required for integration tests")

    db = _database_client_from_env()
    if db.use_memory or db._conn is None:
        pytest.fail("PostgreSQL/pgvector is required; integration test must not silently skip")

    try:
        with db._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute("DROP TABLE IF EXISTS face_embeddings CASCADE;")
            cur.execute("DROP TABLE IF EXISTS persons CASCADE;")
            cur.execute("""
                CREATE TABLE persons (
                    person_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    status TEXT DEFAULT 'active'
                );
            """)
            cur.execute("""
                CREATE TABLE face_embeddings (
                    id BIGSERIAL PRIMARY KEY,
                    person_id TEXT NOT NULL REFERENCES persons(person_id),
                    embedding vector(4) NOT NULL,
                    model_version TEXT NOT NULL,
                    quality_score REAL DEFAULT 1.0,
                    source_image_path TEXT,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
            """)
            cur.execute("CREATE INDEX face_embeddings_hnsw ON face_embeddings USING hnsw (embedding vector_cosine_ops);")

        db.create_person("person_A", "Alice")
        db.create_person("person_B", "Bob")

        # Alice has many templates, all close to the query.
        base_a = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        for _ in range(10):
            db.add_embedding("person_A", base_a, model_version="arcface_v1")

        # Bob is the genuine second candidate and must survive person-level top-k.
        base_b = np.array([0.98, 0.20, 0.0, 0.0], dtype=np.float32)
        db.add_embedding("person_B", base_b, model_version="arcface_v1")

        query = np.array([1.0, 0.02, 0.0, 0.0], dtype=np.float32)
        candidates = db.search_top_k(query, model_version="arcface_v1", top_k=2)

        assert [c.person_id for c in candidates] == ["person_A", "person_B"]
        assert candidates[0].similarity > candidates[1].similarity
    finally:
        with db._conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS face_embeddings CASCADE;")
            cur.execute("DROP TABLE IF EXISTS persons CASCADE;")
        db.close()
