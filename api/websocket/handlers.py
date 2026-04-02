"""
api/websocket/handlers.py
==========================
WebSocket route handlers for real-time data streams.

Each handler connects a client to a specific event room and keeps
the connection alive, relying on the manager to push events.

These are mounted directly on the FastAPI app (not via APIRouter)
because WebSocket routes use a different decorator pattern.

Usage (mount in app.py)
-----------------------
    from api.websocket.handlers import register_websocket_routes
    register_websocket_routes(app)
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request

logger = logging.getLogger(__name__)


def register_websocket_routes(app: FastAPI) -> None:
    """Mount all WebSocket routes on the app."""

    @app.websocket("/ws/training")
    async def ws_training(websocket: WebSocket):
        """Stream live training metrics (reward, loss, CVaR, step count)."""
        manager = websocket.app.state.ws_manager
        await manager.connect(websocket, room="training")
        try:
            while True:
                data = await websocket.receive_text()
                if data == "ping":
                    await websocket.send_text("pong")
        except WebSocketDisconnect:
            manager.disconnect(websocket, room="training")

    @app.websocket("/ws/network")
    async def ws_network(websocket: WebSocket):
        """Stream live network topology updates."""
        manager = websocket.app.state.ws_manager
        await manager.connect(websocket, room="network")
        try:
            while True:
                data = await websocket.receive_text()
                if data == "ping":
                    await websocket.send_text("pong")
        except WebSocketDisconnect:
            manager.disconnect(websocket, room="network")

    @app.websocket("/ws/drift")
    async def ws_drift(websocket: WebSocket):
        """Stream drift score updates and drift event notifications."""
        manager = websocket.app.state.ws_manager
        await manager.connect(websocket, room="drift")
        try:
            while True:
                data = await websocket.receive_text()
                if data == "ping":
                    await websocket.send_text("pong")
        except WebSocketDisconnect:
            manager.disconnect(websocket, room="drift")

    @app.websocket("/ws/game")
    async def ws_game(websocket: WebSocket):
        """Stream Bayesian belief updates and game state changes."""
        manager = websocket.app.state.ws_manager
        await manager.connect(websocket, room="game")
        try:
            while True:
                data = await websocket.receive_text()
                if data == "ping":
                    await websocket.send_text("pong")
        except WebSocketDisconnect:
            manager.disconnect(websocket, room="game")

    @app.websocket("/ws/all")
    async def ws_all(websocket: WebSocket):
        """Subscribe to all events (every room combined)."""
        manager = websocket.app.state.ws_manager
        await manager.connect(websocket, room="all")
        try:
            while True:
                data = await websocket.receive_text()
                if data == "ping":
                    await websocket.send_text("pong")
        except WebSocketDisconnect:
            manager.disconnect(websocket, room="all")

    logger.info("WebSocket routes registered: /ws/{training,network,drift,game,all,alerts}")