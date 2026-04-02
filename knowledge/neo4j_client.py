"""
knowledge/neo4j_client.py
==========================
Neo4j driver wrapper — connection pool, session management, and
a clean execute() interface used by every other knowledge module.

Design decisions
----------------
- Uses the official ``neo4j`` Python driver (async-capable).
- All queries go through ``execute_query()`` so we have one place
  for retry logic, logging, and metrics.
- Connection details come from environment variables / config dict,
  never hardcoded.
- ``with Neo4jClient(...) as client:`` is the recommended usage
  pattern — ensures the driver is properly closed on exit.

Environment variables (override via config dict)
------------------------------------------------
    NEO4J_URI       bolt://localhost:7687
    NEO4J_USER      neo4j
    NEO4J_PASSWORD  password
    NEO4J_DATABASE  neo4j

Usage
-----
    client = Neo4jClient.from_env()

    # Run a read query
    records = client.execute_query(
        "MATCH (c:CVE {id: $cve_id}) RETURN c",
        cve_id="CVE-2021-44228",
        mode="read",
    )

    # Run a write query
    client.execute_query(
        "CREATE (c:CVE {id: $id, description: $desc})",
        id="CVE-2021-44228",
        desc="Log4Shell ...",
        mode="write",
    )

    # Batch write (much faster than individual writes)
    client.execute_batch(
        "UNWIND $rows AS row CREATE (c:CVE {id: row.id})",
        rows=[{"id": "CVE-2021-44228"}, {"id": "CVE-2021-26084"}],
    )

    client.close()
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional

logger = logging.getLogger(__name__)

# Attempt to import the neo4j driver — fail gracefully if not installed
try:
    from neo4j import GraphDatabase, Driver, Session, Result
    from neo4j.exceptions import (
        ServiceUnavailable,
        AuthError,
        ClientError,
        TransientError,
    )
    _NEO4J_AVAILABLE = True
except ImportError:
    _NEO4J_AVAILABLE = False
    logger.warning(
        "neo4j driver not installed. Install with: pip install neo4j. "
        "KG features will be unavailable."
    )


# ── Default connection settings ────────────────────────────────────────────
_DEFAULT_URI      = "bolt://localhost:7687"
_DEFAULT_USER     = "neo4j"
_DEFAULT_PASSWORD = "password"
_DEFAULT_DATABASE = "neo4j"

# Retry settings for transient errors
_MAX_RETRIES    = 3
_RETRY_DELAY_S  = 2.0

# Batch size for UNWIND-based bulk inserts
_DEFAULT_BATCH_SIZE = 500


class Neo4jClient:
    """
    Production-ready Neo4j client with connection pooling and retry logic.

    Parameters
    ----------
    uri : str
        Bolt URI of the Neo4j instance.
    user : str
        Neo4j username.
    password : str
        Neo4j password.
    database : str
        Target database name.  Use ``"neo4j"`` for the default database.
    max_connection_pool_size : int
        Maximum connections in the driver pool.  Default 50.
    connection_timeout : float
        Seconds before a connection attempt times out.  Default 10.
    """

    def __init__(
        self,
        uri:                       str = _DEFAULT_URI,
        user:                      str = _DEFAULT_USER,
        password:                  str = _DEFAULT_PASSWORD,
        database:                  str = _DEFAULT_DATABASE,
        max_connection_pool_size:  int = 50,
        connection_timeout:        float = 10.0,
    ) -> None:
        self.uri      = uri
        self.user     = user
        self.database = database

        self._driver: Optional[Any] = None   # neo4j.Driver

        if not _NEO4J_AVAILABLE:
            logger.warning("Neo4jClient created but neo4j driver is not installed.")
            return

        try:
            self._driver = GraphDatabase.driver(
                uri,
                auth=(user, password),
                max_connection_pool_size=max_connection_pool_size,
                connection_timeout=connection_timeout,
            )
            logger.info("Neo4j driver created — uri=%r, database=%r", uri, database)
        except Exception as exc:
            logger.error("Failed to create Neo4j driver: %s", exc)
            raise

    # ------------------------------------------------------------------ #
    # Factory methods
    # ------------------------------------------------------------------ #

    @classmethod
    def from_env(cls) -> "Neo4jClient":
        """
        Build a client from environment variables.

        Reads:
            NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE
        """
        return cls(
            uri      = os.getenv("NEO4J_URI",      _DEFAULT_URI),
            user     = os.getenv("NEO4J_USER",     _DEFAULT_USER),
            password = os.getenv("NEO4J_PASSWORD", _DEFAULT_PASSWORD),
            database = os.getenv("NEO4J_DATABASE", _DEFAULT_DATABASE),
        )

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Neo4jClient":
        """
        Build a client from a config dict.

        Expected keys: ``uri``, ``user``, ``password``, ``database``.
        Falls back to environment variables for any missing keys.
        """
        neo4j_cfg = config.get("neo4j", config)
        return cls(
            uri      = neo4j_cfg.get("uri",      os.getenv("NEO4J_URI",      _DEFAULT_URI)),
            user     = neo4j_cfg.get("user",     os.getenv("NEO4J_USER",     _DEFAULT_USER)),
            password = neo4j_cfg.get("password", os.getenv("NEO4J_PASSWORD", _DEFAULT_PASSWORD)),
            database = neo4j_cfg.get("database", os.getenv("NEO4J_DATABASE", _DEFAULT_DATABASE)),
            max_connection_pool_size = neo4j_cfg.get("pool_size", 50),
        )

    # ------------------------------------------------------------------ #
    # Core query execution
    # ------------------------------------------------------------------ #

    def execute_query(
        self,
        query:  str,
        mode:   str = "read",
        **params: Any,
    ) -> List[Dict[str, Any]]:
        """
        Execute a Cypher query and return all records as dicts.

        Parameters
        ----------
        query : str
            Cypher query string.  Use ``$param_name`` for parameters.
        mode : str
            ``"read"``  — uses a read transaction (can be load-balanced).
            ``"write"`` — uses a write transaction (always routed to primary).
        **params
            Query parameters (passed as keyword arguments for clarity).

        Returns
        -------
        list[dict]
            Each record in the result set as a plain Python dict.
            Returns empty list if neo4j is not installed.

        Raises
        ------
        RuntimeError
            If the driver is not connected.
        ClientError
            For query syntax / schema errors (not retried).
        """
        if not _NEO4J_AVAILABLE or self._driver is None:
            logger.debug("Neo4j unavailable — returning empty result for query.")
            return []

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                with self._session() as session:
                    if mode == "write":
                        result = session.execute_write(
                            lambda tx: list(tx.run(query, **params))
                        )
                    else:
                        result = session.execute_read(
                            lambda tx: list(tx.run(query, **params))
                        )

                return [dict(record) for record in result]

            except TransientError as exc:
                if attempt < _MAX_RETRIES:
                    logger.warning(
                        "Transient Neo4j error (attempt %d/%d): %s — retrying in %.1fs",
                        attempt, _MAX_RETRIES, exc, _RETRY_DELAY_S,
                    )
                    time.sleep(_RETRY_DELAY_S * attempt)
                else:
                    logger.error("Neo4j query failed after %d attempts: %s", _MAX_RETRIES, exc)
                    raise

            except ClientError as exc:
                # Syntax / schema errors — don't retry
                logger.error("Neo4j client error (query): %s\nQuery: %s", exc, query)
                raise

    def execute_batch(
        self,
        query:      str,
        rows:       List[Dict[str, Any]],
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ) -> int:
        """
        Execute a parameterised batch write using UNWIND.

        Splits ``rows`` into chunks of ``batch_size`` and commits each
        chunk as a separate transaction — avoids loading millions of rows
        into a single transaction which can exhaust Neo4j heap memory.

        Parameters
        ----------
        query : str
            Cypher query using ``$rows`` as the UNWIND parameter.
            Example: ``"UNWIND $rows AS row MERGE (c:CVE {id: row.id})"``
        rows : list[dict]
            List of parameter dicts, one per row to write.
        batch_size : int
            Rows per transaction.  Default 500.

        Returns
        -------
        int
            Total number of rows processed.
        """
        if not _NEO4J_AVAILABLE or self._driver is None:
            logger.debug("Neo4j unavailable — skipping batch write.")
            return 0

        total = 0
        n_batches = (len(rows) + batch_size - 1) // batch_size

        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            batch_num = i // batch_size + 1

            self.execute_query(query, mode="write", rows=batch)
            total += len(batch)

            logger.debug(
                "Batch %d/%d written (%d rows)", batch_num, n_batches, len(batch)
            )

        logger.info("Batch write complete — %d rows in %d batches", total, n_batches)
        return total

    # ------------------------------------------------------------------ #
    # Schema management
    # ------------------------------------------------------------------ #

    def create_constraints(self) -> None:
        """
        Create uniqueness constraints and indexes on the KG schema.

        Idempotent — safe to call multiple times.
        Constraints also create implicit indexes, so no separate
        index creation is needed for constrained properties.
        """
        constraints = [
            # Uniqueness constraints (also create indexes)
            "CREATE CONSTRAINT cve_id_unique IF NOT EXISTS "
            "FOR (c:CVE) REQUIRE c.id IS UNIQUE",

            "CREATE CONSTRAINT technique_id_unique IF NOT EXISTS "
            "FOR (t:Technique) REQUIRE t.technique_id IS UNIQUE",

            "CREATE CONSTRAINT tactic_id_unique IF NOT EXISTS "
            "FOR (t:Tactic) REQUIRE t.tactic_id IS UNIQUE",

            "CREATE CONSTRAINT host_name_unique IF NOT EXISTS "
            "FOR (h:Host) REQUIRE h.name IS UNIQUE",

            # Full-text index for CVE description search
            "CREATE FULLTEXT INDEX cve_description_ft IF NOT EXISTS "
            "FOR (c:CVE) ON EACH [c.description, c.id]",

            # Full-text index for technique name/description search
            "CREATE FULLTEXT INDEX technique_ft IF NOT EXISTS "
            "FOR (t:Technique) ON EACH [t.name, t.description]",
        ]

        for stmt in constraints:
            try:
                self.execute_query(stmt, mode="write")
                logger.debug("Applied: %s", stmt[:60] + "...")
            except Exception as exc:
                # Constraint already exists — not an error
                if "already exists" in str(exc).lower():
                    logger.debug("Constraint already exists, skipping.")
                else:
                    logger.warning("Constraint error: %s", exc)

        logger.info("Neo4j schema constraints verified.")

    def drop_all(self) -> None:
        """
        Delete ALL nodes and relationships in the database.

        .. warning::
            Destructive operation — only call during development/testing
            or before a full KG rebuild.
        """
        self.execute_query("MATCH (n) DETACH DELETE n", mode="write")
        logger.warning("All nodes and relationships deleted from Neo4j database.")

    # ------------------------------------------------------------------ #
    # Health check
    # ------------------------------------------------------------------ #

    def ping(self) -> bool:
        """
        Test connectivity to the Neo4j instance.

        Returns
        -------
        bool
            True if the database is reachable and accepting queries.
        """
        if not _NEO4J_AVAILABLE or self._driver is None:
            return False
        try:
            self._driver.verify_connectivity()
            return True
        except Exception as exc:
            logger.warning("Neo4j ping failed: %s", exc)
            return False

    def get_stats(self) -> Dict[str, Any]:
        """
        Return basic statistics about the current KG contents.

        Returns
        -------
        dict
            Keys: n_cves, n_techniques, n_tactics, n_hosts,
            n_exploits_edges, n_uses_edges.
        """
        queries = {
            "n_cves":          "MATCH (c:CVE) RETURN count(c) AS n",
            "n_techniques":    "MATCH (t:Technique) RETURN count(t) AS n",
            "n_tactics":       "MATCH (t:Tactic) RETURN count(t) AS n",
            "n_hosts":         "MATCH (h:Host) RETURN count(h) AS n",
            "n_exploits":      "MATCH ()-[:EXPLOITS]->() RETURN count(*) AS n",
            "n_uses":          "MATCH ()-[:USES]->() RETURN count(*) AS n",
            "n_maps_to":       "MATCH ()-[:MAPS_TO]->() RETURN count(*) AS n",
        }
        stats: Dict[str, Any] = {}
        for key, q in queries.items():
            try:
                result = self.execute_query(q, mode="read")
                stats[key] = result[0]["n"] if result else 0
            except Exception:
                stats[key] = -1

        return stats

    # ------------------------------------------------------------------ #
    # Context manager
    # ------------------------------------------------------------------ #

    def close(self) -> None:
        """Close the driver and release all pooled connections."""
        if self._driver is not None:
            self._driver.close()
            self._driver = None
            logger.info("Neo4j driver closed.")

    def __enter__(self) -> "Neo4jClient":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    @contextmanager
    def _session(self) -> Generator[Any, None, None]:
        """Open a Neo4j session and ensure it is closed after use."""
        session = self._driver.session(database=self.database)
        try:
            yield session
        finally:
            session.close()

    def __repr__(self) -> str:
        status = "connected" if self._driver is not None else "disconnected"
        return (
            f"Neo4jClient(uri={self.uri!r}, "
            f"database={self.database!r}, "
            f"status={status})"
        )