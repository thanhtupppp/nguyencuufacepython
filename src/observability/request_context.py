"""Request correlation and safe structured logging context."""
from __future__ import annotations

import logging
import re
from contextvars import ContextVar
from datetime import datetime, timezone
from uuid import uuid4

_REQUEST_ID = ContextVar("request_id", default=None)
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


def new_request_id(candidate: str | None = None) -> str:
    """Validate a client request id or create a bounded server-generated id."""
    value = (candidate or "").strip()
    if value and _REQUEST_ID_RE.fullmatch(value):
        return value
    return uuid4().hex


def set_request_id(request_id: str) -> None:
    if not _REQUEST_ID_RE.fullmatch(request_id):
        raise ValueError("invalid request id")
    _REQUEST_ID.set(request_id)


def get_request_id() -> str:
    value = _REQUEST_ID.get()
    return value if isinstance(value, str) else ""


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


class RequestContextFilter(logging.Filter):
    """Attach correlation data without logging credentials or biometric payloads."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id() or "-"
        return True
