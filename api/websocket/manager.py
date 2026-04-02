"""
api/websocket/manager.py
=========================
WebSocket connection pool with per-room broadcasting.

Rooms allow clients to subscribe to specific event streams:
    "alerts"    — live alert feed
    "training"  — training progress metrics
    "network"   — network topology updates
    "drift"     — drift score updates
    "game"      — game model / belief updates

Clients in "all" (default) receive every event.

Usage
-----
    manager = WebSocketManager()

    # In route handler
    await manager.connect(websocket, room="alerts")

    # Broadcast to all clients in a room
    await manager.broadcast_json({"event": "alert", ...}, room="alerts")

    # Broadcast to all clients (every room)
    await manager.broadcast_json({"event": "tick"})
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set

from fastapi import WebSocket
from starlette.websockets import WebSocketState

logger = logging.getLogger(__name__)


class WebSocketManager:
    """
    Manages WebSocket connections organised into rooms.

    Thread-safe for async use (single event loop).

    Parameters
    ----------
    max_connections : int
        Maximum simultaneous WebSocket connections.  Default 100.
    """

    def __init__(self, max_connections: int = 100) -> None:
        self.max_connections = max_connections

        # room name → set of active WebSocket connections
        self._rooms:       Dict[str, Set[WebSocket]] = defaultdict(set)
        self._all_sockets: Set[WebSocket]            = set()

    # ------------------------------------------------------------------ #
    # Connection lifecycle
    # ------------------------------------------------------------------ #

    async def connect(
        self,
        websocket: WebSocket,
        room:      str = "all",
    ) -> None:
        """
        Accept and register a WebSocket connection.

        Parameters
        ----------
        websocket : WebSocket
        room : str
            Room to subscribe to.  Default ``"all"``.
        """
        if len(self._all_sockets) >= self.max_connections:
            await websocket.close(code=1013, reason="Server at capacity.")
            return

        await websocket.accept()
        self._rooms[room].add(websocket)
        self._all_sockets.add(websocket)

        logger.debug(
            "WebSocket connected — room=%r, total=%d",
            room, len(self._all_sockets),
        )

        # Send a welcome message
        try:
            await websocket.send_json({
                "event": "connected",
                "room":  room,
                "message": f"Connected to ACD Framework WebSocket ({room} room).",
            })
        except Exception:
            pass

    def disconnect(
        self,
        websocket: WebSocket,
        room:      Optional[str] = None,
    ) -> None:
        """
        Remove a WebSocket from the connection pool.

        Parameters
        ----------
        websocket : WebSocket
        room : str, optional
            Room to remove from.  If None, removes from all rooms.
        """
        self._all_sockets.discard(websocket)

        if room:
            self._rooms[room].discard(websocket)
        else:
            for room_sockets in self._rooms.values():
                room_sockets.discard(websocket)

        logger.debug(
            "WebSocket disconnected — total=%d", len(self._all_sockets)
        )

    # ------------------------------------------------------------------ #
    # Broadcasting
    # ------------------------------------------------------------------ #

    async def broadcast_json(
        self,
        data: Dict[str, Any],
        room: Optional[str] = None,
    ) -> int:
        """
        Broadcast a JSON message to all connections in a room.

        Parameters
        ----------
        data : dict
            JSON-serialisable payload.
        room : str, optional
            Target room.  If None, broadcasts to ALL connections.

        Returns
        -------
        int
            Number of clients the message was sent to.
        """
        if room is not None:
            targets = list(self._rooms.get(room, set()))
        else:
            targets = list(self._all_sockets)

        if not targets:
            return 0

        sent  = 0
        dead: List[WebSocket] = []

        for ws in targets:
            try:
                if ws.client_state == WebSocketState.CONNECTED:
                    await ws.send_json(data)
                    sent += 1
                else:
                    dead.append(ws)
            except Exception as exc:
                logger.debug("WebSocket send failed: %s", exc)
                dead.append(ws)

        # Clean up dead connections
        for ws in dead:
            self.disconnect(ws)

        return sent

    async def broadcast_text(
        self,
        text: str,
        room: Optional[str] = None,
    ) -> int:
        """Broadcast a raw text message."""
        if room is not None:
            targets = list(self._rooms.get(room, set()))
        else:
            targets = list(self._all_sockets)

        sent = 0
        dead = []

        for ws in targets:
            try:
                if ws.client_state == WebSocketState.CONNECTED:
                    await ws.send_text(text)
                    sent += 1
                else:
                    dead.append(ws)
            except Exception:
                dead.append(ws)

        for ws in dead:
            self.disconnect(ws)

        return sent

    async def send_to(
        self,
        websocket: WebSocket,
        data:      Dict[str, Any],
    ) -> bool:
        """
        Send a JSON message to a single WebSocket connection.

        Returns
        -------
        bool
            True if sent successfully.
        """
        try:
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_json(data)
                return True
        except Exception as exc:
            logger.debug("Direct send failed: %s", exc)
            self.disconnect(websocket)
        return False

    # ------------------------------------------------------------------ #
    # Event helpers (typed broadcasts)
    # ------------------------------------------------------------------ #

    async def emit_alert(self, alert: Dict[str, Any]) -> None:
        """Broadcast an alert to the 'alerts' room and 'all'."""
        payload = {"event": "alert", **alert}
        await self.broadcast_json(payload, room="alerts")
        await self.broadcast_json(payload, room="all")

    async def emit_training_update(self, metrics: Dict[str, Any]) -> None:
        """Broadcast training metrics to the 'training' room."""
        await self.broadcast_json({"event": "training_update", **metrics}, room="training")

    async def emit_drift_event(self, drift_event: Dict[str, Any]) -> None:
        """Broadcast a drift event to 'drift' and 'alerts' rooms."""
        payload = {"event": "drift_detected", **drift_event}
        await self.broadcast_json(payload, room="drift")
        await self.broadcast_json(payload, room="alerts")

    async def emit_network_update(self, state: Dict[str, Any]) -> None:
        """Broadcast a network state update to 'network' room."""
        await self.broadcast_json({"event": "network_update", **state}, room="network")

    async def emit_belief_update(self, belief: Dict[str, Any]) -> None:
        """Broadcast a belief update to the 'game' room."""
        await self.broadcast_json({"event": "belief_update", **belief}, room="game")

    # ------------------------------------------------------------------ #
    # Stats
    # ------------------------------------------------------------------ #

    @property
    def n_connections(self) -> int:
        """Total active WebSocket connections."""
        return len(self._all_sockets)

    def get_room_counts(self) -> Dict[str, int]:
        """Return connection counts per room."""
        return {room: len(sockets) for room, sockets in self._rooms.items()}

    def __repr__(self) -> str:
        return (
            f"WebSocketManager("
            f"connections={self.n_connections}, "
            f"rooms={list(self._rooms.keys())})"
        )