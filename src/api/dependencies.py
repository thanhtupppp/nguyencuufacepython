import os
import secrets
from typing import Optional

from fastapi import HTTPException, Query, Security, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

from src.database.client import DatabaseClient
from src.api.recognition_runtime import build_recognition_pipeline

from pathlib import Path

# One shared client per API process keeps identity and embedding operations in
# the same database connection/store.
sqlite_path_env = os.getenv("SQLITE_PATH")
if not sqlite_path_env and not os.getenv("DATABASE_URL") and Path("data/faces.db").exists():
    sqlite_path_env = "data/faces.db"

db = DatabaseClient(sqlite_path=sqlite_path_env) if sqlite_path_env else DatabaseClient()

# Recognition is intentionally initialized once per API process. Missing or
# invalid real model assets leave this as None, so face endpoints fail closed.
try:
    recognition_pipeline = build_recognition_pipeline()
except (RuntimeError, FileNotFoundError, ValueError):
    recognition_pipeline = None

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
bearer_scheme = HTTPBearer(auto_error=False)


def verify_api_key(
    header_key: Optional[str] = Security(api_key_header),
    bearer: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme),
    query_key: Optional[str] = Query(None, alias="api_key"),
) -> bool:
    """Verifies API Key from X-API-Key header, Authorization: Bearer, or ?api_key= query.

    If API_KEY is configured in the environment, requests must provide matching
    credentials. If unconfigured (development mode), requests pass through.
    """
    expected_key = os.getenv("API_KEY", "").strip()
    if not expected_key:
        return True

    provided_key = header_key or (bearer.credentials if bearer else None) or query_key
    if not provided_key or not secrets.compare_digest(provided_key, expected_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return True
