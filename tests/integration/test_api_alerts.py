"""
tests/integration/test_api_alerts.py
======================================
Integration tests for the alerts API endpoints.
"""

from __future__ import annotations

from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def app():
    from api.app import create_app
    application = create_app()

    # Seed with sample alerts
    application.state.alerts = [
        {"id": "a1", "severity": "CRITICAL", "title": "Op_Server0 breach", "source": "drift",
         "acknowledged": False, "timestamp": "2026-03-15T10:00:00Z"},
        {"id": "a2", "severity": "HIGH", "title": "Malicious process", "source": "agent",
         "acknowledged": False, "timestamp": "2026-03-15T10:01:00Z"},
        {"id": "a3", "severity": "LOW", "title": "Routine scan", "source": "system",
         "acknowledged": True, "timestamp": "2026-03-15T10:02:00Z"},
    ]
    application.state.ws_manager = MagicMock()

    return application


@pytest.fixture
def client(app):
    return TestClient(app)


class TestAlertEndpoints:

    def test_list_alerts(self, client) -> None:
        """GET /alerts should return list of alerts."""
        resp = client.get("/alerts/")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)

    def test_list_alerts_filter_severity(self, client) -> None:
        """GET /alerts?severity=CRITICAL filters by severity."""
        resp = client.get("/alerts/?severity=CRITICAL")
        assert resp.status_code == 200
        data = resp.json()
        assert all(a["severity"] == "CRITICAL" for a in data)

    def test_get_single_alert(self, client) -> None:
        """GET /alerts/{id} should return the alert."""
        resp = client.get("/alerts/a1")
        assert resp.status_code == 200
        assert resp.json()["id"] == "a1"

    def test_get_missing_alert_returns_404(self, client) -> None:
        """GET /alerts/{id} for non-existent alert should return 404."""
        resp = client.get("/alerts/nonexistent")
        assert resp.status_code == 404

    def test_acknowledge_alert(self, client) -> None:
        """PATCH /alerts/{id} should update alert fields."""
        resp = client.patch("/alerts/a1", json={"acknowledged": True})
        assert resp.status_code == 200

    def test_alert_stats(self, client) -> None:
        """GET /alerts/stats/summary should return severity counts."""
        resp = client.get("/alerts/stats/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert "total" in data
        assert "by_severity" in data
