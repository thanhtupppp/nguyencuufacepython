"""FastAPI transport boundary with explicit lifecycle and readiness contracts."""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Callable

from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.event_repository import EventRepository, InMemoryEventRepository, PostgresEventRepository
from app.fastapi_contract import RecognitionEvent
from src.contracts.error import error_envelope
from src.observability.request_context import get_request_id, new_request_id, reset_request_id, set_request_id
from src.health.runtime import RuntimeLifecycle


def create_app(
    repository: EventRepository,
    *,
    readiness: RuntimeLifecycle | None = None,
    dependency_probes: dict[str, Callable[[], tuple[bool, str]]] | None = None,
) -> FastAPI:
    lifecycle = readiness or RuntimeLifecycle()
    probes = dependency_probes or {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.lifecycle = lifecycle
        lifecycle.started = True
        try:
            for name, probe in probes.items():
                try:
                    ok, detail = probe()
                except Exception:
                    ok, detail = False, "probe_failed"
                if ok:
                    lifecycle.mark_ready(name, detail)
                else:
                    lifecycle.mark_unready(name, detail)
            lifecycle.startup_complete()
            yield
        finally:
            lifecycle.shutdown()

    app = FastAPI(title="Face Recognition API", version="0.3.0", lifespan=lifespan)

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = new_request_id(request.headers.get("X-Request-ID"))
        token = set_request_id(request_id)
        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            reset_request_id(token)

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=error_envelope("VALIDATION_ERROR", "Request validation failed"),
            headers={"X-Request-ID": get_request_id() or new_request_id()},
        )

    @app.exception_handler(Exception)
    async def unexpected_exception_handler(request: Request, exc: Exception):
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=error_envelope("INTERNAL_ERROR", "Internal server error"),
            headers={"X-Request-ID": get_request_id() or new_request_id()},
        )

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz(response: Response) -> dict[str, object]:
        snapshot = lifecycle.gate.as_dict()
        if not lifecycle.gate.ready:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return snapshot

    @app.post("/v1/events")
    def ingest_event(event: RecognitionEvent):
        request_id = get_request_id() or new_request_id()
        try:
            accepted, payload = repository.put_if_absent(event.event_id, event.model_dump(mode="json"))
        except Exception:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content=error_envelope("DEPENDENCY_UNAVAILABLE", "Event persistence is unavailable"),
            )
        return JSONResponse(
            status_code=202 if accepted else 200,
            content={"status": "accepted" if accepted else "duplicate", "event": payload, "request_id": request_id},
        )

    @app.websocket("/v1/events/ws")
    async def events_ws(websocket: WebSocket):
        request_id = new_request_id(websocket.headers.get("X-Request-ID"))
        token = set_request_id(request_id)
        await websocket.accept(headers=[(b"X-Request-ID", request_id.encode("ascii"))])
        try:
            while True:
                event = RecognitionEvent.model_validate_json(await websocket.receive_text())
                accepted, payload = repository.put_if_absent(event.event_id, event.model_dump(mode="json"))
                await websocket.send_json({
                    "status": "accepted" if accepted else "duplicate",
                    "request_id": request_id,
                    "timestamp": payload.get("timestamp"),
                    "event": payload,
                })
        except WebSocketDisconnect:
            return
        except RequestValidationError:
            await websocket.send_json(error_envelope("VALIDATION_ERROR", "Invalid event payload"))
        except ValueError:
            await websocket.send_json(error_envelope("VALIDATION_ERROR", "Invalid event payload"))
        except Exception:
            await websocket.send_json(error_envelope("DEPENDENCY_UNAVAILABLE", "Event persistence is unavailable"))
        finally:
            reset_request_id(token)

    return app


def _production_repository() -> EventRepository:
    """Create the production repository only after explicit server startup configuration."""
    database_url = os.getenv("DATABASE_URL", "").strip()
    if not database_url:
        raise RuntimeError("DATABASE_URL must be configured for production startup")
    return PostgresEventRepository(database_url)


def create_production_app() -> FastAPI:
    """Build the production ASGI app; never falls back to the in-memory repository."""
    repository = _production_repository()
    from src.api.main import build_production_probes
    lifecycle = RuntimeLifecycle()
    probes = build_production_probes(repository, lifecycle)
    return create_app(repository, readiness=lifecycle, dependency_probes=probes)


# Import-safe test/inspection app. Production must use create_production_app().
app = create_app(InMemoryEventRepository())
