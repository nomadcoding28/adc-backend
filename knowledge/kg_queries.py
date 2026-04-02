"""
knowledge/kg_queries.py
========================
Named Cypher queries for all KG API routes and dashboard features.

Design principles
-----------------
- NO inline Cypher strings anywhere else in the codebase.
  All queries live here so they are easy to audit, test, and optimise.
- Every public method returns plain Python dicts (no neo4j Record objects).
- Query parameters are always passed as keyword arguments — never by
  string interpolation (prevents Cypher injection).

Usage
-----
    from knowledge import KGQuerier, Neo4jClient

    client  = Neo4jClient.from_env()
    querier = KGQuerier(client)

    # Get the full graph for the D3 force visualisation
    graph = querier.get_full_graph(limit=200)

    # Get the attack chain for a specific CVE
    chain = querier.get_attack_chain("CVE-2021-44228")

    # Search nodes by text
    results = querier.search("Log4Shell")

    # Get CVE details
    cve = querier.get_cve("CVE-2021-44228")
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from knowledge.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


class KGQuerier:
    """
    High-level query interface for the ACD knowledge graph.

    All methods return JSON-serialisable Python dicts/lists.
    Results are suitable for direct return from FastAPI route handlers.

    Parameters
    ----------
    client : Neo4jClient
        Connected Neo4j client.
    """

    def __init__(self, client: Neo4jClient) -> None:
        self.client = client

    # ------------------------------------------------------------------ #
    # Full graph (for D3 force visualisation on the dashboard)
    # ------------------------------------------------------------------ #

    def get_full_graph(
        self,
        limit:          int = 200,
        min_cvss:       float = 0.0,
        tactic_filter:  Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Return all nodes and edges for the D3 force-directed graph viewer.

        Parameters
        ----------
        limit : int
            Maximum number of nodes to return.  Default 200.
        min_cvss : float
            Only include CVEs with at least this CVSS score.
        tactic_filter : str, optional
            If given, only return subgraph for this tactic ID (e.g. "TA0001").

        Returns
        -------
        dict
            ``{"nodes": [...], "edges": [...]}``.
            Each node: ``{id, type, label, properties}``.
            Each edge: ``{source, target, type, properties}``.
        """
        # ── Nodes ──────────────────────────────────────────────────────
        node_query = (
            "MATCH (n) WHERE "
            "(n:CVE AND n.max_cvss >= $min_cvss) OR "
            "n:Technique OR n:Tactic OR n:Host "
            "RETURN n, labels(n) AS labels "
            f"LIMIT {limit}"
        )
        node_rows = self.client.execute_query(
            node_query, mode="read", min_cvss=min_cvss
        )
        nodes = [self._format_node(r["n"], r["labels"]) for r in node_rows]

        # ── Edges ──────────────────────────────────────────────────────
        edge_query = (
            "MATCH (a)-[r]->(b) "
            "WHERE (a:CVE AND a.max_cvss >= $min_cvss) OR a:Technique OR a:Tactic "
            "RETURN "
            "  CASE WHEN a:CVE THEN a.id "
            "       WHEN a:Technique THEN a.technique_id "
            "       WHEN a:Tactic THEN a.tactic_id "
            "       ELSE a.name END AS source_id, "
            "  CASE WHEN b:CVE THEN b.id "
            "       WHEN b:Technique THEN b.technique_id "
            "       WHEN b:Tactic THEN b.tactic_id "
            "       ELSE b.name END AS target_id, "
            "  type(r) AS rel_type, "
            "  properties(r) AS props "
            f"LIMIT {limit * 3}"
        )
        edge_rows = self.client.execute_query(
            edge_query, mode="read", min_cvss=min_cvss
        )
        edges = [
            {
                "source":     r["source_id"],
                "target":     r["target_id"],
                "type":       r["rel_type"],
                "properties": dict(r.get("props") or {}),
            }
            for r in edge_rows
        ]

        return {"nodes": nodes, "edges": edges}

    # ------------------------------------------------------------------ #
    # CVE queries
    # ------------------------------------------------------------------ #

    def get_cve(self, cve_id: str) -> Optional[Dict[str, Any]]:
        """
        Return full details for a single CVE node.

        Parameters
        ----------
        cve_id : str
            CVE identifier, e.g. ``"CVE-2021-44228"``.

        Returns
        -------
        dict or None
        """
        rows = self.client.execute_query(
            "MATCH (c:CVE {id: $cve_id}) "
            "OPTIONAL MATCH (c)-[r:MAPS_TO]->(t:Technique) "
            "OPTIONAL MATCH (c)-[:EXPLOITS]->(h:Host) "
            "RETURN c, "
            "  collect(DISTINCT {technique_id: t.technique_id, name: t.name, score: r.score}) AS techniques, "
            "  collect(DISTINCT h.name) AS hosts",
            cve_id=cve_id,
        )
        if not rows:
            return None

        row    = rows[0]
        cve    = dict(row["c"])
        result = {
            **cve,
            "techniques": [t for t in row["techniques"] if t.get("technique_id")],
            "hosts":       row["hosts"],
        }
        return result

    def get_cves_for_host(self, host_name: str) -> List[Dict[str, Any]]:
        """
        Return all CVEs that target a specific host.

        Parameters
        ----------
        host_name : str
            CybORG host name (e.g. ``"User0"``).

        Returns
        -------
        list[dict]
        """
        rows = self.client.execute_query(
            "MATCH (c:CVE)-[:EXPLOITS]->(h:Host {name: $host}) "
            "RETURN c.id AS id, c.description AS description, "
            "       c.max_cvss AS max_cvss, c.severity AS severity "
            "ORDER BY c.max_cvss DESC",
            host=host_name,
        )
        return [dict(r) for r in rows]

    def get_top_cves(
        self, limit: int = 20, min_cvss: float = 7.0
    ) -> List[Dict[str, Any]]:
        """
        Return the top N highest-severity CVEs in the graph.

        Parameters
        ----------
        limit : int
            Number of results.
        min_cvss : float
            Minimum CVSS filter.

        Returns
        -------
        list[dict]
        """
        rows = self.client.execute_query(
            "MATCH (c:CVE) "
            "WHERE c.max_cvss >= $min_cvss "
            "RETURN c.id AS id, c.description AS description, "
            "       c.max_cvss AS max_cvss, c.severity AS severity, "
            "       c.published AS published "
            "ORDER BY c.max_cvss DESC "
            f"LIMIT {limit}",
            min_cvss=min_cvss,
        )
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------ #
    # Technique queries
    # ------------------------------------------------------------------ #

    def get_technique(self, technique_id: str) -> Optional[Dict[str, Any]]:
        """
        Return full details for a single ATT&CK technique.

        Parameters
        ----------
        technique_id : str
            ATT&CK technique ID, e.g. ``"T1190"``.

        Returns
        -------
        dict or None
        """
        rows = self.client.execute_query(
            "MATCH (t:Technique {technique_id: $tid}) "
            "OPTIONAL MATCH (t)-[:BELONGS_TO]->(ta:Tactic) "
            "OPTIONAL MATCH (c:CVE)-[:MAPS_TO]->(t) "
            "RETURN t, "
            "  collect(DISTINCT {tactic_id: ta.tactic_id, name: ta.name}) AS tactics, "
            "  collect(DISTINCT {id: c.id, max_cvss: c.max_cvss}) AS cves",
            tid=technique_id,
        )
        if not rows:
            return None

        row = rows[0]
        return {
            **dict(row["t"]),
            "tactics": [ta for ta in row["tactics"] if ta.get("tactic_id")],
            "cves":    [c  for c  in row["cves"]    if c.get("id")],
        }

    def get_techniques_for_tactic(
        self, tactic_id: str
    ) -> List[Dict[str, Any]]:
        """Return all techniques belonging to a given tactic."""
        rows = self.client.execute_query(
            "MATCH (t:Technique)-[:BELONGS_TO]->(ta:Tactic {tactic_id: $tid}) "
            "RETURN t.technique_id AS technique_id, t.name AS name, "
            "       t.description AS description, t.platforms AS platforms "
            "ORDER BY t.technique_id",
            tid=tactic_id,
        )
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------ #
    # Attack chain (for the incident detail view)
    # ------------------------------------------------------------------ #

    def get_attack_chain(self, cve_id: str) -> Dict[str, Any]:
        """
        Return the predicted attack kill chain for a given CVE.

        Traverses: CVE → Techniques → Tactics, ordered by kill-chain stage.

        Parameters
        ----------
        cve_id : str
            Starting CVE identifier.

        Returns
        -------
        dict
            ``{"cve_id": ..., "steps": [...]}``.
            Each step: ``{stage, technique_id, name, tactic_id, tactic_name, score}``.
        """
        rows = self.client.execute_query(
            "MATCH (c:CVE {id: $cve_id})-[r:MAPS_TO]->(t:Technique) "
            "OPTIONAL MATCH (t)-[:BELONGS_TO]->(ta:Tactic) "
            "RETURN t.technique_id AS technique_id, t.name AS name, "
            "       ta.tactic_id AS tactic_id, ta.name AS tactic_name, "
            "       r.score AS score "
            "ORDER BY r.score DESC",
            cve_id=cve_id,
        )

        steps = []
        for i, row in enumerate(rows, start=1):
            steps.append({
                "stage":         i,
                "technique_id":  row.get("technique_id"),
                "name":          row.get("name"),
                "tactic_id":     row.get("tactic_id"),
                "tactic_name":   row.get("tactic_name"),
                "score":         round(float(row.get("score") or 0.0), 4),
            })

        return {"cve_id": cve_id, "steps": steps}

    def get_predicted_next_technique(
        self, active_technique_ids: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Predict the next likely ATT&CK technique given the currently
        observed techniques.

        Uses a co-occurrence heuristic: finds techniques that share
        the same tactic as the current observed techniques.

        Parameters
        ----------
        active_technique_ids : list[str]
            Technique IDs currently detected in the environment.

        Returns
        -------
        list[dict]
            Predicted next techniques, ordered by likelihood score.
        """
        if not active_technique_ids:
            return []

        rows = self.client.execute_query(
            "MATCH (active:Technique)-[:BELONGS_TO]->(ta:Tactic) "
            "WHERE active.technique_id IN $active_ids "
            "MATCH (next:Technique)-[:BELONGS_TO]->(ta2:Tactic) "
            "WHERE NOT next.technique_id IN $active_ids "
            "  AND ta2.tactic_id > ta.tactic_id "   # next kill-chain stage
            "RETURN DISTINCT next.technique_id AS technique_id, "
            "       next.name AS name, ta2.tactic_id AS tactic_id, "
            "       ta2.name AS tactic_name, count(*) AS co_count "
            "ORDER BY co_count DESC LIMIT 10",
            active_ids=active_technique_ids,
        )
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------ #
    # Search
    # ------------------------------------------------------------------ #

    def search(
        self,
        query:     str,
        node_type: Optional[str] = None,
        limit:     int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Full-text search across CVEs, Techniques, and Tactics.

        Parameters
        ----------
        query : str
            Search string.
        node_type : str, optional
            Filter to a specific node type: ``"CVE"``, ``"Technique"``,
            ``"Tactic"``.  None = all types.
        limit : int
            Maximum results to return.

        Returns
        -------
        list[dict]
            Each result: ``{id, type, label, score, properties}``.
        """
        results: List[Dict[str, Any]] = []
        q_lower  = query.lower()

        # CVE search (full-text index)
        if node_type in (None, "CVE"):
            rows = self.client.execute_query(
                "CALL db.index.fulltext.queryNodes('cve_description_ft', $query) "
                "YIELD node, score "
                f"RETURN node.id AS id, 'CVE' AS type, node.id AS label, "
                "       score, node.max_cvss AS cvss, node.severity AS severity "
                f"LIMIT {limit}",
                query=query,
            )
            for r in rows:
                results.append({
                    "id":         r["id"],
                    "type":       "CVE",
                    "label":      r["label"],
                    "score":      round(float(r.get("score") or 0.0), 4),
                    "properties": {"max_cvss": r.get("cvss"), "severity": r.get("severity")},
                })

        # Technique search
        if node_type in (None, "Technique"):
            rows = self.client.execute_query(
                "CALL db.index.fulltext.queryNodes('technique_ft', $query) "
                "YIELD node, score "
                "RETURN node.technique_id AS id, 'Technique' AS type, "
                "       node.name AS label, score "
                f"LIMIT {limit}",
                query=query,
            )
            for r in rows:
                results.append({
                    "id":         r["id"],
                    "type":       "Technique",
                    "label":      r["label"],
                    "score":      round(float(r.get("score") or 0.0), 4),
                    "properties": {},
                })

        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    # ------------------------------------------------------------------ #
    # Statistics
    # ------------------------------------------------------------------ #

    def get_stats(self) -> Dict[str, Any]:
        """
        Return summary statistics about the knowledge graph.

        Returns
        -------
        dict
            Keys: n_cves, n_techniques, n_tactics, n_hosts,
            n_maps_to, n_exploits, n_belongs_to.
        """
        return self.client.get_stats()

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _format_node(
        node: Any, labels: List[str]
    ) -> Dict[str, Any]:
        """Convert a neo4j Node object to a JSON-serialisable dict."""
        props = dict(node) if node else {}
        label = labels[0] if labels else "Unknown"

        # Pick a display ID based on node type
        node_id = (
            props.get("id")           # CVE
            or props.get("technique_id")  # Technique
            or props.get("tactic_id")     # Tactic
            or props.get("name")          # Host
            or "unknown"
        )

        display_label = props.get("name") or node_id

        return {
            "id":         node_id,
            "type":       label,
            "label":      display_label,
            "properties": {k: v for k, v in props.items() if k != "description"},
        }

    def __repr__(self) -> str:
        return f"KGQuerier(client={self.client!r})"