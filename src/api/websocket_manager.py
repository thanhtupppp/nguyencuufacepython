"""
WebSocket connection manager for real-time event broadcasting.
Maintains active WebSocket connections and broadcasts recognition events to clients.
"""

import asyncio
import logging
from typing import Any
from fastapi import WebSocket

logger = logging.getLogger("api.websocket")


class ConnectionManager:
    """Manages active WebSocket subscribers for real-time access events."""

    def __init__(self) -> None:
        self.active_connections: list[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        """Accepts and registers a new WebSocket client."""
        await websocket.accept()
        async with self._lock:
            self.active_connections.append(websocket)
        logger.info(f"WebSocket client connected. Total active: {len(self.active_connections)}")

    async def disconnect(self, websocket: WebSocket) -> None:
        """Unregisters a disconnected WebSocket client."""
        async with self._lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
        logger.info(f"WebSocket client disconnected. Total active: {len(self.active_connections)}")

    async def send_personal_message(self, message: dict[str, Any], websocket: WebSocket) -> None:
        """Sends a JSON message to a single specific client."""
        try:
            await websocket.send_json(message)
        except Exception as exc:
            logger.warning(f"Failed to send personal message to WebSocket client: {exc}")

    async def broadcast(self, message: dict[str, Any]) -> None:
        """
        Broadcasts a JSON event payload to all currently connected clients.
        Automatically cleans up any broken or closed connections.
        """
        if not self.active_connections:
            return

        dead_connections: list[WebSocket] = []
        async with self._lock:
            targets = list(self.active_connections)

        for connection in targets:
            try:
                await connection.send_json(message)
            except Exception as exc:
                logger.warning(f"Failed to broadcast to client, queueing for removal: {exc}")
                dead_connections.append(connection)

        if dead_connections:
            async with self._lock:
                for dead in dead_connections:
                    if dead in self.active_connections:
                        self.active_connections.remove(dead)


# Global singleton instance for the FastAPI application process
ws_manager = ConnectionManager()
