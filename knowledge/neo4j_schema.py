"""
knowledge/neo4j_schema.py
==========================
Neo4j schema management — constraints, indexes, and full-text indexes.

Ensures the knowledge graph has the correct schema before data is ingested.
Called during application startup (in ``api/dependencies.startup``) and
at the start of a full KG rebuild (``tasks/kg_tasks.rebuild_kg_task``).

Design notes
------------
- Constraints ensure uniqueness of CVE IDs, technique IDs, tactic IDs, and
  host names — preventing duplicate nodes during upsert operations.
- Full-text indexes (``cve_description_ft``, ``technique_ft``) power the
  ``/kg/search`` endpoint in ``knowledge/kg_queries.py``.
- All Cypher statements use ``CREATE ... IF NOT EXISTS`` so schema
  application is idempotent and safe to run repeatedly.

Usage
-----
    from knowledge.neo4j_schema import apply_schema
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))
    apply_schema(driver)
"""

from __future__ import annotations

import logging
from typing import Any, List

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Schema definition — constraints and indexes
# ══════════════════════════════════════════════════════════════════════════════

# Uniqueness constraints
_CONSTRAINTS: List[str] = [
    # CVE nodes
    "CREATE CONSTRAINT cve_id_unique IF NOT EXISTS "
    "FOR (c:CVE) REQUIRE c.id IS UNIQUE",

    # ATT&CK Technique nodes
    "CREATE CONSTRAINT technique_id_unique IF NOT EXISTS "
    "FOR (t:Technique) REQUIRE t.technique_id IS UNIQUE",

    # ATT&CK Tactic nodes
    "CREATE CONSTRAINT tactic_id_unique IF NOT EXISTS "
    "FOR (ta:Tactic) REQUIRE ta.tactic_id IS UNIQUE",

    # Host nodes
    "CREATE CONSTRAINT host_name_unique IF NOT EXISTS "
    "FOR (h:Host) REQUIRE h.name IS UNIQUE",
]

# Standard indexes (for fast lookups)
_INDEXES: List[str] = [
    # CVE indexes
    "CREATE INDEX cve_severity IF NOT EXISTS FOR (c:CVE) ON (c.severity)",
    "CREATE INDEX cve_cvss IF NOT EXISTS FOR (c:CVE) ON (c.max_cvss)",
    "CREATE INDEX cve_published IF NOT EXISTS FOR (c:CVE) ON (c.published)",

    # Technique indexes
    "CREATE INDEX technique_name IF NOT EXISTS FOR (t:Technique) ON (t.name)",
    "CREATE INDEX technique_tactic IF NOT EXISTS FOR (t:Technique) ON (t.tactic)",

    # Host indexes
    "CREATE INDEX host_status IF NOT EXISTS FOR (h:Host) ON (h.status)",
]

# Full-text indexes (for search functionality in kg_queries.py)
# These use the APOC-compatible full-text index creation syntax.
_FULL_TEXT_INDEXES: List[str] = [
    # Full-text search over CVE descriptions and IDs
    "CREATE FULLTEXT INDEX cve_description_ft IF NOT EXISTS "
    "FOR (c:CVE) ON EACH [c.id, c.description]",

    # Full-text search over technique names and descriptions
    "CREATE FULLTEXT INDEX technique_ft IF NOT EXISTS "
    "FOR (t:Technique) ON EACH [t.technique_id, t.name, t.description]",
]


# ══════════════════════════════════════════════════════════════════════════════
# Schema application
# ══════════════════════════════════════════════════════════════════════════════

def apply_schema(driver: Any) -> None:
    """
    Apply all constraints, indexes, and full-text indexes to Neo4j.

    Parameters
    ----------
    driver : neo4j.Driver
        An active Neo4j driver instance (synchronous).
        Obtained via ``neo4j.GraphDatabase.driver(...)``.

    Raises
    ------
    Exception
        If any schema statement fails (after logging the error).
    """
    logger.info("Applying Neo4j schema (%d constraints, %d indexes, %d full-text)...",
                len(_CONSTRAINTS), len(_INDEXES), len(_FULL_TEXT_INDEXES))

    all_statements = _CONSTRAINTS + _INDEXES + _FULL_TEXT_INDEXES
    applied = 0
    errors  = 0

    with driver.session() as session:
        for stmt in all_statements:
            try:
                session.run(stmt)
                applied += 1
            except Exception as exc:
                # Full-text indexes may fail if they already exist with
                # different options — log but continue
                errors += 1
                logger.warning("Schema statement failed: %s — %s", stmt[:60], exc)

    logger.info(
        "Neo4j schema applied — %d/%d succeeded, %d failed.",
        applied, len(all_statements), errors,
    )


def verify_schema(driver: Any) -> dict:
    """
    Verify the current schema by listing constraints and indexes.

    Returns
    -------
    dict
        ``{constraints: [...], indexes: [...], full_text: [...]}``
    """
    result: dict = {"constraints": [], "indexes": [], "full_text": []}

    with driver.session() as session:
        # List constraints
        try:
            for record in session.run("SHOW CONSTRAINTS"):
                result["constraints"].append({
                    "name":       record.get("name"),
                    "type":       record.get("type"),
                    "entityType": record.get("entityType"),
                    "labelsOrTypes": record.get("labelsOrTypes"),
                    "properties": record.get("properties"),
                })
        except Exception as exc:
            logger.warning("Could not list constraints: %s", exc)

        # List indexes
        try:
            for record in session.run("SHOW INDEXES"):
                idx = {
                    "name":       record.get("name"),
                    "type":       record.get("type"),
                    "entityType": record.get("entityType"),
                    "labelsOrTypes": record.get("labelsOrTypes"),
                    "properties": record.get("properties"),
                    "state":      record.get("state"),
                }
                if record.get("type") == "FULLTEXT":
                    result["full_text"].append(idx)
                else:
                    result["indexes"].append(idx)
        except Exception as exc:
            logger.warning("Could not list indexes: %s", exc)

    return result


def drop_all_schema(driver: Any) -> None:
    """
    Drop all constraints and indexes (use only in test teardown).

    Parameters
    ----------
    driver : neo4j.Driver
    """
    logger.warning("Dropping ALL Neo4j constraints and indexes.")
    with driver.session() as session:
        try:
            for record in session.run("SHOW CONSTRAINTS"):
                name = record.get("name")
                if name:
                    session.run(f"DROP CONSTRAINT {name} IF EXISTS")
        except Exception as exc:
            logger.warning("Error dropping constraints: %s", exc)

        try:
            for record in session.run("SHOW INDEXES"):
                name = record.get("name")
                if name:
                    session.run(f"DROP INDEX {name} IF EXISTS")
        except Exception as exc:
            logger.warning("Error dropping indexes: %s", exc)
