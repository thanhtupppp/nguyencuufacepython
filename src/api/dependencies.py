import os
import secrets
from pathlib import Path
from typing import Optional

from fastapi import HTTPException, Query, Security, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

from src.database import DatabaseClient
from src.api.recognition_runtime import build_recognition_pipeline

# Tests may inject SQLite/memory, but production initialization is explicit.
sqlite_path_env = os.getenv("SQLITE_PATH")
if not sqlite_path_env and not os.getenv("DATABASE_URL") and Path("data/faces.db").exists():
    sqlite_path_env = "data/faces.db"

db = DatabaseClient(sqlite_path=sqlite_path_env) if sqlite_path_env else DatabaseClient()

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
