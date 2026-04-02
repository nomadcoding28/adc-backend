"""
tests/integration/test_api_ws.py
=================================
Integration tests for WebSocket connections.

The WebSocket manager is initialised during the app lifespan (startup).
FastAPI TestClient must be used as a context manager to trigger the lifespan.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a test client with lifespan so ws_manager is initialised."""
    from api.app import create_app
    application = create_app()

    # TestClient must be used as a context manager to trigger lifespan
    # (startup/shutdown), which initialises ws_manager, alerts, etc.
    with TestClient(application) as c:
        yield c


class TestWebSocketEndpoints:

    def test_alerts_ws_connect(self, client) -> None:
        """WebSocket /alerts/ws/alerts should accept connection and send welcome."""
        with client.websocket_connect("/alerts/ws/alerts") as ws:
            # First message is the welcome JSON from WebSocketManager.connect()
            welcome = ws.receive_json()
            assert welcome["event"] == "connected"
            assert welcome["room"] == "alerts"
            assert "message" in welcome

    def test_ws_ping_pong(self, client) -> None:
        """WebSocket should respond with 'pong' to 'ping'."""
        with client.websocket_connect("/alerts/ws/alerts") as ws:
            # Consume the welcome message first
            _ = ws.receive_json()

            # Now test ping/pong
            ws.send_text("ping")
            data = ws.receive_text()
            assert data == "pong"
