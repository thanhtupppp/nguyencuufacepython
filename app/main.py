"""FastAPI transport boundary for trusted recognition events."""
from __future__ import annotations

import os

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from app.event_repository import EventRepository, InMemoryEventRepository, PostgresEventRepository
from app.fastapi_contract import RecognitionEvent


def create_app(repository: EventRepository) -> FastAPI:
    app = FastAPI(title="Face Recognition API", version="0.1.0")

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

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
app = create_app(PostgresEventRepository(_database_url))
