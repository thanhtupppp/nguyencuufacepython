"""Shared service dependencies for API routes."""

from src.database.client import DatabaseClient
from src.api.recognition_runtime import build_recognition_pipeline

# One shared client per API process keeps identity and embedding operations in
# the same database connection/store. Production deployment can replace this
# with a dependency container or pool without changing route contracts.
db = DatabaseClient()

# Recognition is intentionally initialized once per API process. Missing or
# invalid real model assets leave this as None, so face endpoints fail closed.
try:
    recognition_pipeline = build_recognition_pipeline()
except (RuntimeError, FileNotFoundError, ValueError):
    recognition_pipeline = None
