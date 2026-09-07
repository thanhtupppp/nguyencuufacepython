"""
Event routing: WebSocket real-time subscription and Edge event publishing.
"""

from datetime import datetime, timezone
import os
import secrets
from typing import Any, Optional

from fastapi import APIRouter, Depends, Query, WebSocket, WebSocketDisconnect, status
from pydantic import BaseModel, Field

from src.api.dependencies import db, verify_api_key
from src.api.websocket_manager import ws_manager

router = APIRouter()


class AccessEventPayload(BaseModel):
    """Event payload submitted by edge cameras / video analytic workers."""
    event_type: str = Field(default="ACCESS_EVENT", description="Type of event, e.g. ACCESS_EVENT, SECURITY_ALERT")
    person_id: Optional[str] = Field(default=None, description="Enrolled person ID if matched")
    name: Optional[str] = Field(default=None, description="Human readable name if matched")
    device_id: Optional[str] = Field(default="default_cam", description="Camera or access terminal identifier")
    similarity: Optional[float] = Field(default=None, description="Cosine similarity score")
    margin: Optional[float] = Field(default=None, description="Margin difference between top1 and top2 candidates")
    status: str = Field(default="MATCHED", description="Recognition status: MATCHED, UNKNOWN, AMBIGUOUS, MASK_DETECTED, SPOOF_DETECTED")
    liveness: Optional[str] = Field(default="PASS", description="Liveness decision: PASS, FAIL, INCONCLUSIVE")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary additional metadata")


@router.websocket("/ws/v1/events")
async def websocket_events_endpoint(
    websocket: WebSocket,
    api_key: Optional[str] = Query(default=None),
) -> None:
    """
    Real-time WebSocket event stream for access control and face recognition monitoring.
    Clients connect to receive live JSON event broadcasts whenever an access event occurs.
    """
    # Verify API key if configured
    expected_key = os.getenv("API_KEY", "").strip()
    if expected_key:
        if not api_key or not secrets.compare_digest(api_key, expected_key):
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return

    await ws_manager.connect(websocket)
    try:
        # Initial greeting handshake
        await websocket.send_json({
            "event_type": "CONNECTED",
            "message": "Subscribed to NguyenCuuFacePython real-time access events.",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        while True:
            # Keep-alive / heartbeat loop
            data = await websocket.receive_text()
            if data.strip().lower() == "ping":
                await websocket.send_json({
                    "event_type": "PONG",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
    except WebSocketDisconnect:
        await ws_manager.disconnect(websocket)
    except Exception:
        await ws_manager.disconnect(websocket)


@router.post("/api/v1/events", dependencies=[Depends(verify_api_key)], status_code=201)
async def publish_event(payload: AccessEventPayload) -> dict[str, Any]:
    """
    Allows external cameras, edge devices (e.g. ESP32, RTSP runners) to push an event.
    The event is recorded in the database access log and broadcast immediately to all WebSocket subscribers.
    """
    event_dict = payload.model_dump()
    event_dict["timestamp"] = datetime.now(timezone.utc).isoformat()

    # Log access to database
    db.log_access(
        person_id=payload.person_id,
        device_id=payload.device_id,
        similarity=payload.similarity,
        margin=payload.margin,
        status=payload.status,
    )

    # Broadcast event to WebSocket subscribers
    await ws_manager.broadcast(event_dict)

    return {
        "status": "ok",
        "broadcasted_to": len(ws_manager.active_connections),
        "event": event_dict,
    }
