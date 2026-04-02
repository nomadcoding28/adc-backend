"""
tasks/kg_tasks.py
==================
Celery tasks for Knowledge Graph maintenance.

Tasks
-----
    update_cves(days)
        Incremental NVD CVE ingest (daily beat schedule).

    rebuild_kg_task()
        Full KG rebuild — re-ingest all CVEs + ATT&CK data.

    ingest_attck()
        Parse MITRE ATT&CK STIX bundle and upsert into Neo4j.

    rebuild_rag_index()
        Re-embed all KG documents and rebuild the FAISS index.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

from celery import Task
from celery.exceptions import SoftTimeLimitExceeded

from tasks.celery_app import celery_app

logger = logging.getLogger(__name__)


class _BaseKGTask(Task):
    """Base task that ensures Neo4j driver cleanup on exit."""

    abstract = True

    def __init__(self) -> None:
        super().__init__()
        self._client = None

    def after_return(self, status, retval, task_id, args, kwargs, einfo) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None

    def _get_client(self, config: Dict[str, Any]) -> Any:
        """Lazily create a Neo4j client for this task."""
        if self._client is None:
            from knowledge.neo4j_client import Neo4jClient
            self._client = Neo4jClient.from_config(config)
        return self._client


# ══════════════════════════════════════════════════════════════════════════════
# Incremental CVE update (daily)
# ══════════════════════════════════════════════════════════════════════════════

@celery_app.task(
    bind          = True,
    base          = _BaseKGTask,
    name          = "tasks.kg_tasks.update_cves",
    queue         = "kg",
    max_retries   = 2,
    track_started = True,
)
def update_cves(
    self:   Task,
    config: Optional[Dict[str, Any]] = None,
    days:   int = 1,
) -> Dict[str, Any]:
    """
    Incremental CVE ingest from NVD.

    Fetches CVEs modified in the last ``days`` days and upserts
    them into the Neo4j knowledge graph.

    Parameters
    ----------
    config : dict, optional
        Application config.  If None, loads from config.yaml.
    days : int
        Look-back window in days.  Default 1 (yesterday).

    Returns
    -------
    dict
        ``{status, cves_upserted, elapsed_s}``
    """
    start_time = time.monotonic()
    cfg = config or _load_config()

    logger.info("CVE update START — days=%d", days)

    try:
        client = self._get_client(cfg)
        from knowledge.kg_builder import KGBuilder
        builder = KGBuilder(client=client, config=cfg.get("knowledge_graph", {}))

        count = builder.ingest_nvd(days=days)
        elapsed = round(time.monotonic() - start_time, 1)

        logger.info("CVE update COMPLETE — upserted=%d  elapsed=%.1fs", count, elapsed)
        return {
            "status":        "completed",
            "cves_upserted": count,
            "elapsed_s":     elapsed,
        }

    except SoftTimeLimitExceeded:
        logger.warning("CVE update time limit reached.")
        return {"status": "time_limit"}
    except Exception as exc:
        logger.error("CVE update FAILED — %s", exc, exc_info=True)
        raise self.retry(exc=exc, countdown=60)


# ══════════════════════════════════════════════════════════════════════════════
# Full KG rebuild
# ══════════════════════════════════════════════════════════════════════════════

@celery_app.task(
    bind          = True,
    base          = _BaseKGTask,
    name          = "tasks.kg_tasks.rebuild_kg",
    queue         = "kg",
    max_retries   = 0,
    track_started = True,
    ignore_result = False,
)
def rebuild_kg_task(
    self:   Task,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Full KG rebuild — clear and re-ingest all data.

    Steps:
        1. Apply schema (constraints + indexes)
        2. Ingest all CVEs from NVD
        3. Ingest ATT&CK techniques
        4. Link CVEs → Techniques → Tactics
    """
    start_time = time.monotonic()
    cfg = config or _load_config()
    run_id = (self.request.id or "local")[:8]

    logger.info("KG rebuild START — run_id=%s", run_id)
    self.update_state(state="STARTED", meta={"run_id": run_id, "message": "Applying schema..."})

    try:
        client = self._get_client(cfg)

        # Step 1: Apply schema
        from knowledge.neo4j_schema import apply_schema
        apply_schema(client.driver)
        logger.info("Schema applied.")

        # Step 2: Build KG
        from knowledge.kg_builder import KGBuilder
        builder = KGBuilder(client=client, config=cfg.get("knowledge_graph", {}))

        self.update_state(
            state="PROGRESS",
            meta={"run_id": run_id, "message": "Ingesting CVEs from NVD..."},
        )
        cve_count = builder.ingest_nvd(days=365)

        self.update_state(
            state="PROGRESS",
            meta={"run_id": run_id, "message": "Ingesting ATT&CK techniques..."},
        )
        technique_count = builder.ingest_attck()

        self.update_state(
            state="PROGRESS",
            meta={"run_id": run_id, "message": "Linking CVEs to techniques..."},
        )
        link_count = builder.link_cves_to_techniques()

        elapsed = round(time.monotonic() - start_time, 1)
        logger.info(
            "KG rebuild COMPLETE — CVEs=%d  Techniques=%d  Links=%d  elapsed=%.1fs",
            cve_count, technique_count, link_count, elapsed,
        )

        return {
            "status":          "completed",
            "run_id":          run_id,
            "cves_ingested":   cve_count,
            "techniques":      technique_count,
            "links_created":   link_count,
            "elapsed_s":       elapsed,
        }

    except SoftTimeLimitExceeded:
        logger.warning("KG rebuild time limit reached.")
        return {"status": "time_limit", "run_id": run_id}
    except Exception as exc:
        logger.error("KG rebuild FAILED — %s", exc, exc_info=True)
        raise


# ══════════════════════════════════════════════════════════════════════════════
# ATT&CK ingest
# ══════════════════════════════════════════════════════════════════════════════

@celery_app.task(
    bind          = True,
    base          = _BaseKGTask,
    name          = "tasks.kg_tasks.ingest_attck",
    queue         = "kg",
    max_retries   = 1,
    track_started = True,
)
def ingest_attck(
    self:   Task,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Parse the MITRE ATT&CK Enterprise STIX bundle and upsert into Neo4j.

    Returns
    -------
    dict
        ``{status, techniques_upserted, tactics_upserted, elapsed_s}``
    """
    start_time = time.monotonic()
    cfg = config or _load_config()

    logger.info("ATT&CK ingest START")

    try:
        client  = self._get_client(cfg)
        from knowledge.kg_builder import KGBuilder
        builder = KGBuilder(client=client, config=cfg.get("knowledge_graph", {}))

        technique_count = builder.ingest_attck()
        elapsed = round(time.monotonic() - start_time, 1)

        logger.info("ATT&CK ingest COMPLETE — techniques=%d  elapsed=%.1fs",
                     technique_count, elapsed)
        return {
            "status":              "completed",
            "techniques_upserted": technique_count,
            "elapsed_s":           elapsed,
        }

    except Exception as exc:
        logger.error("ATT&CK ingest FAILED — %s", exc, exc_info=True)
        raise self.retry(exc=exc, countdown=30)


# ══════════════════════════════════════════════════════════════════════════════
# RAG index rebuild
# ══════════════════════════════════════════════════════════════════════════════

@celery_app.task(
    bind          = True,
    base          = _BaseKGTask,
    name          = "tasks.kg_tasks.rebuild_rag_index",
    queue         = "kg",
    max_retries   = 0,
    track_started = True,
)
def rebuild_rag_index(
    self:   Task,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Re-embed all KG documents and rebuild the FAISS index.

    Returns
    -------
    dict
        ``{status, n_indexed, elapsed_s}``
    """
    start_time = time.monotonic()
    cfg = config or _load_config()

    logger.info("RAG index rebuild START")

    try:
        from explainability.rag.document_store import DocumentStore
        from explainability.rag.embedder import Embedder
        from explainability.rag.indexer import FAISSIndexer

        store    = DocumentStore()
        embedder = Embedder()
        indexer  = FAISSIndexer(
            store=store, embedder=embedder, config=cfg.get("rag", {}),
        )

        retriever  = indexer.rebuild()
        n_indexed  = retriever.n_indexed if retriever else 0
        elapsed    = round(time.monotonic() - start_time, 1)

        logger.info("RAG index rebuild COMPLETE — n_indexed=%d  elapsed=%.1fs",
                     n_indexed, elapsed)
        return {
            "status":    "completed",
            "n_indexed": n_indexed,
            "elapsed_s": elapsed,
        }

    except Exception as exc:
        logger.error("RAG index rebuild FAILED — %s", exc, exc_info=True)
        raise


# ── Helper ───────────────────────────────────────────────────────────────────

def _load_config() -> Dict[str, Any]:
    """Load configuration from config.yaml."""
    try:
        from utils.config_loader import load_config
        return load_config("config.yaml")
    except Exception as exc:
        logger.warning("Could not load config.yaml: %s", exc)
        return {}
