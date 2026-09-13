"""FastAPI transport boundary with explicit lifecycle and readiness contracts."""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Callable

from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect, status
from fastapi.responses import JSONResponse

from app.event_repository import EventRepository, InMemoryEventRepository, PostgresEventRepository
from app.fastapi_contract import RecognitionEvent
from src.contracts.error import error_envelope
from src.observability.request_context import get_request_id, new_request_id, set_request_id
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

    app = FastAPI(title="Face Recognition API", version="0.2.0", lifespan=lifespan)

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = new_request_id(request.headers.get("X-Request-ID"))
        set_request_id(request_id)
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

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
        set_request_id(request_id)
        await websocket.accept(headers=[(b"X-Request-ID", request_id.encode("ascii"))])
        try:
            while True:
                event = RecognitionEvent.model_validate_json(await websocket.receive_text())
                accepted, payload = repository.put_if_absent(
                    event.event_id, event.model_dump(mode="json")
                )
                await websocket.send_json(
                    {
                        "status": "accepted" if accepted else "duplicate",
                        "request_id": request_id,
                        "timestamp": payload.get("timestamp"),
                        "event": payload,
                    }
                )
        except WebSocketDisconnect:
            return
        except ValueError:
            await websocket.send_json(error_envelope("VALIDATION_ERROR", "Invalid event payload"))
        except Exception:
            await websocket.send_json(error_envelope("DEPENDENCY_UNAVAILABLE", "Event persistence is unavailable"))

    return app


def _production_repository() -> EventRepository:
    """Create the production repository on explicit application startup."""
    database_url = os.getenv("DATABASE_URL", "").strip()
    if not database_url:
        raise RuntimeError("DATABASE_URL must be configured when starting the production server")
    return PostgresEventRepository(database_url)


# Conventional ASGI export for test/inspection imports; it performs no network IO.
app = create_app(InMemoryEventRepository())
