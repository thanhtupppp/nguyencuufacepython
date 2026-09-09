"""Minimal FastAPI transport boundary for trusted recognition events."""
from __future__ import annotations

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from app.fastapi_contract import RecognitionEvent

app = FastAPI(title="Face Recognition API", version="0.1.0")
_seen_events: dict[str, dict] = {}


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/events")
def ingest_event(event: RecognitionEvent):
    event_key = str(event.event_id)
    if event_key in _seen_events:
        return JSONResponse(
            status_code=200,
            content={"status": "duplicate", "event": _seen_events[event_key]},
        )

    payload = event.model_dump(mode="json")
    _seen_events[event_key] = payload
    return JSONResponse(status_code=202, content={"status": "accepted", "event": payload})


@app.websocket("/v1/events/ws")
async def events_ws(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            event = RecognitionEvent.model_validate_json(await websocket.receive_text())
            event_key = str(event.event_id)
            if event_key in _seen_events:
                await websocket.send_json(
                    {"status": "duplicate", "event": _seen_events[event_key]}
                )
                continue
            payload = event.model_dump(mode="json")
            _seen_events[event_key] = payload
            await websocket.send_json({"status": "accepted", "event": payload})
    except WebSocketDisconnect:
        return
