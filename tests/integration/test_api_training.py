"""
tests/integration/test_api_training.py
=======================================
Integration tests for the training API endpoints.

Uses FastAPI TestClient — no actual Celery/Redis needed.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def app():
    """Create a test FastAPI app with mocked dependencies."""
    from api.app import create_app

    application = create_app()

    # Mock app.state
    application.state.agent = MagicMock()
    application.state.agent.is_training = False
    application.state.agent.get_metrics.return_value = {
        "mean_reward": 42.0, "cvar_005": -15.0, "episode_count": 100,
    }
    application.state.env = MagicMock()
    application.state.training_task = None

    return application


@pytest.fixture
def client(app):
    return TestClient(app)


class TestTrainingEndpoints:
    """Integration tests for /training/* endpoints."""

    def test_get_training_status(self, client) -> None:
        """GET /training/status should return current status."""
        resp = client.get("/training/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "is_training" in data

    def test_get_training_config(self, client) -> None:
        """GET /training/config should return training configuration."""
        resp = client.get("/training/config")
        assert resp.status_code == 200

    def test_start_training_returns_task(self, client) -> None:
        """POST /training/start should accept a training request."""
        resp = client.post("/training/start", json={
            "agent_type": "cvar_ppo",
            "total_timesteps": 10000,
        })
        # May return 200 or 202 depending on Celery availability
        assert resp.status_code in (200, 202, 409), (
            f"Unexpected status {resp.status_code}: {resp.text}"
        )

    def test_stop_training_when_not_running(self, client) -> None:
        """DELETE /training/stop when not training may return 200 or 409."""
        resp = client.delete("/training/stop")
        assert resp.status_code in (200, 409)
