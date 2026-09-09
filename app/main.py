"""FastAPI transport boundary for trusted recognition events."""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from app.event_repository import EventRepository, PostgresEventRepository
from app.fastapi_contract import RecognitionEvent
from app.model_registry import ModelRegistryError, RecognitionModelRegistry


def create_app(repository: EventRepository, model_registry: RecognitionModelRegistry | None = None) -> FastAPI:
    app = FastAPI(title="Face Recognition API", version="0.1.0")
    app.state.model_registry = model_registry

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        registry = app.state.model_registry
        if registry is None:
            return {"status": "ok", "recognition_models": "not_configured"}
        return {
            "status": "ok" if registry.approved else "degraded",
            "recognition_models": "approved" if registry.approved else "blocked",
        }

    @app.get("/v1/models/status")
    def model_status():
        registry = app.state.model_registry
        if registry is None:
            return JSONResponse(status_code=503, content={"status": "not_configured"})
        if not registry.approved:
            return JSONResponse(status_code=503, content={"status": "blocked"})
        return {"status": "approved", "receipt": registry.receipt}

    @app.post("/v1/events")
    def ingest_event(event: RecognitionEvent):
        accepted, payload = repository.put_if_absent(event.event_id, event.model_dump(mode="json"))
        return JSONResponse(
            status_code=202 if accepted else 200,
            content={"status": "accepted" if accepted else "duplicate", "event": payload},
        )

    @app.websocket("/v1/events/ws")
    async def events_ws(websocket: WebSocket):
        await websocket.accept()
        try:
            while True:
                event = RecognitionEvent.model_validate_json(await websocket.receive_text())
                accepted, payload = repository.put_if_absent(
                    event.event_id, event.model_dump(mode="json")
                )
                await websocket.send_json(
                    {"status": "accepted" if accepted else "duplicate", "event": payload}
                )
        except WebSocketDisconnect:
            return

    return app


# Production requires DATABASE_URL; tests explicitly inject InMemoryEventRepository.
_database_url = os.getenv("DATABASE_URL")
if not _database_url:
    raise RuntimeError("DATABASE_URL must be set for the production application")

_manifest = os.getenv("RECOGNITION_MODEL_MANIFEST")
_registry: RecognitionModelRegistry | None = None
if _manifest:
    _registry = RecognitionModelRegistry(Path(_manifest))
    try:
        _registry.approve()
    except ModelRegistryError as exc:
        # Keep the API transport alive for health/diagnostics, but never expose
        # an unapproved recognition model through the registry.
        print(f"recognition model registry blocked: {exc}")

app = create_app(PostgresEventRepository(_database_url), _registry)
