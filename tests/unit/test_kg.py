"""
tests/unit/test_kg.py
======================
Unit tests for Knowledge Graph query operations.

Uses a mock Neo4j client — no actual Neo4j connection required.
"""

from __future__ import annotations

from unittest.mock import MagicMock
import pytest


class TestKGQuerier:
    """Test the KGQuerier against a mocked Neo4j client."""

    @pytest.fixture
    def querier(self, mock_kg_client):
        from knowledge.kg_queries import KGQuerier
        return KGQuerier(client=mock_kg_client)

    def test_get_full_graph_empty(self, querier, mock_kg_client) -> None:
        """Empty KG should return empty nodes and edges."""
        mock_kg_client.execute_query.return_value = []
        result = querier.get_full_graph(limit=10)
        assert "nodes" in result
        assert "edges" in result
        assert result["nodes"] == []

    def test_get_cve_not_found(self, querier, mock_kg_client) -> None:
        """Non-existent CVE should return None."""
        mock_kg_client.execute_query.return_value = []
        result = querier.get_cve("CVE-9999-9999")
        assert result is None

    def test_get_cve_found(self, querier, mock_kg_client) -> None:
        """Existing CVE should return a dict with expected keys."""
        mock_kg_client.execute_query.return_value = [{
            "c": {"id": "CVE-2021-44228", "description": "Log4Shell", "max_cvss": 10.0},
            "techniques": [{"technique_id": "T1190", "name": "Exploit Public-Facing App", "score": 0.9}],
            "hosts": ["Enterprise0"],
        }]
        result = querier.get_cve("CVE-2021-44228")
        assert result is not None
        assert result["id"] == "CVE-2021-44228"

    def test_get_attack_chain_empty(self, querier, mock_kg_client) -> None:
        """Attack chain for unknown CVE should return empty steps."""
        mock_kg_client.execute_query.return_value = []
        result = querier.get_attack_chain("CVE-9999-9999")
        assert result["cve_id"] == "CVE-9999-9999"
        assert result["steps"] == []

    def test_get_technique_not_found(self, querier, mock_kg_client) -> None:
        """Non-existent technique should return None."""
        mock_kg_client.execute_query.return_value = []
        result = querier.get_technique("T9999")
        assert result is None

    def test_search_returns_list(self, querier, mock_kg_client) -> None:
        """Search should return a list (even if empty)."""
        mock_kg_client.execute_query.return_value = []
        result = querier.search("Log4Shell")
        assert isinstance(result, list)

    def test_get_stats(self, querier, mock_kg_client) -> None:
        """Stats should return a dict with count keys."""
        result = mock_kg_client.get_stats()
        assert "n_cves" in result
        assert "n_techniques" in result


class TestNeo4jSchema:
    """Test the Neo4j schema management module."""

    def test_apply_schema_runs_without_error(self) -> None:
        """apply_schema should execute without raising (mocked driver)."""
        from knowledge.neo4j_schema import apply_schema

        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)
        mock_session.run.return_value = None

        apply_schema(mock_driver)
        assert mock_session.run.call_count > 0, "Should have executed schema statements."

    def test_verify_schema_returns_dict(self) -> None:
        """verify_schema should return a dict with expected keys."""
        from knowledge.neo4j_schema import verify_schema

        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)
        mock_session.run.return_value = []

        result = verify_schema(mock_driver)
        assert "constraints" in result
        assert "indexes" in result
        assert "full_text" in result
