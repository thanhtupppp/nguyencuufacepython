from types import SimpleNamespace

from src.database.readiness import probe_postgres_pgvector


def test_memory_fallback_is_not_ready():
    ready, detail = probe_postgres_pgvector(SimpleNamespace(use_memory=True, use_sqlite=False, _conn=None))
    assert not ready
    assert detail == "memory_fallback_active"


def test_sqlite_is_not_ready():
    ready, detail = probe_postgres_pgvector(SimpleNamespace(use_memory=False, use_sqlite=True, _conn=None))
    assert not ready
    assert detail == "sqlite_backend_active"


def test_missing_connection_is_not_ready():
    ready, detail = probe_postgres_pgvector(SimpleNamespace(use_memory=False, use_sqlite=False, _conn=None))
    assert not ready
    assert detail == "postgres_connection_missing"


def test_probe_requires_pgvector_and_tables():
    class Cursor:
        def __init__(self):
            self.calls = 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def execute(self, sql, params=None):
            self.calls += 1

        def fetchone(self):
            if self.calls == 1:
                return (1,)
            return ("0.8.0",)

        def fetchall(self):
            return [("persons",), ("face_embeddings",)]

    class Conn:
        def cursor(self):
            return Cursor()

    db = SimpleNamespace(use_memory=False, use_sqlite=False, _conn=Conn())
    ready, detail = probe_postgres_pgvector(db)
    assert ready
    assert detail == "postgres_pgvector_verified:0.8.0"
