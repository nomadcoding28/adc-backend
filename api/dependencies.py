"""
api/dependencies.py
====================
Shared FastAPI dependency injection providers.

All route handlers that need access to the agent, environment, KG client,
RAG retriever, etc. declare them as function parameters with ``Depends()``.

This single file is the wiring hub — it knows how to construct every
shared object and ensures they are singletons (constructed once, reused).

Singleton pattern
-----------------
All heavy objects (agent, KG client, RAG index) are stored on
``app.state`` during startup and retrieved via dependency functions.
This avoids constructing a new agent on every HTTP request.

Usage in route handlers
-----------------------
    from fastapi import Depends
    from api.dependencies import get_agent, get_kg_client

    @router.get("/status")
    async def status(agent = Depends(get_agent)):
        return agent.get_metrics()
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import Depends, HTTPException, Request, status

logger = logging.getLogger(__name__)

# ── Type aliases ────────────────────────────────────────────────────────────
# These avoid circular imports — actual types resolved at runtime
AgentType        = Any
EnvType          = Any
Neo4jClientType  = Any
RAGRetrieverType = Any
DriftDetectorType= Any
GameType         = Any
ReportGenType    = Any
WebSocketMgrType = Any


# ══════════════════════════════════════════════════════════════════════════════
# Application lifecycle
# ══════════════════════════════════════════════════════════════════════════════

async def startup(app: Any) -> None:
    """
    Initialise all shared resources and attach them to app.state.

    Called from the FastAPI lifespan context manager.
    """
    cfg = getattr(app.state, "config", {})

    logger.info("Initialising shared application resources...")

    # ── Neo4j client ───────────────────────────────────────────────────
    try:
        from knowledge.neo4j_client import Neo4jClient
        app.state.kg_client = Neo4jClient.from_config(cfg)
        logger.info("Neo4j client initialised.")
    except Exception as exc:
        logger.warning("Neo4j client failed: %s — KG features disabled.", exc)
        app.state.kg_client = None

    # ── RAG index ──────────────────────────────────────────────────────
    try:
        from explainability.rag.document_store import DocumentStore
        from explainability.rag.embedder import Embedder
        from explainability.rag.indexer import FAISSIndexer

        store    = DocumentStore()
        embedder = Embedder()
        indexer  = FAISSIndexer(store=store, embedder=embedder,
                                config=cfg.get("rag", {}))
        app.state.rag_retriever = indexer.load_or_build()
        logger.info("RAG index loaded.")
    except Exception as exc:
        logger.warning("RAG index failed: %s — explanations disabled.", exc)
        app.state.rag_retriever = None

    # ── LLM client ─────────────────────────────────────────────────────
    try:
        from explainability.llm.client import LLMClient
        app.state.llm_client = LLMClient.from_config(cfg.get("llm", {}))
        logger.info("LLM client initialised.")
    except Exception as exc:
        logger.warning("LLM client failed: %s.", exc)
        app.state.llm_client = None

    # ── Agent (lazy — not loaded until first /training/start call) ─────
    app.state.agent          = None
    app.state.env            = None
    app.state.training_task  = None

    # ── WebSocket manager ──────────────────────────────────────────────
    from api.websocket.manager import WebSocketManager
    app.state.ws_manager = WebSocketManager()
    logger.info("WebSocket manager initialised.")

    # ── Alert store (in-memory) ────────────────────────────────────────
    app.state.alerts = []

    # ── Drift detector ─────────────────────────────────────────────────
    try:
        from drift.detector_factory import DetectorFactory
        app.state.drift_detector = DetectorFactory.build(cfg.get("drift", {}))
        logger.info("Drift detector initialised.")
    except Exception as exc:
        logger.warning("Drift detector failed: %s.", exc)
        app.state.drift_detector = None

    # ── Game model ─────────────────────────────────────────────────────
    try:
        from game.stochastic_game import StochasticGame
        from game.attacker_model import AttackerModel
        from game.belief_updater import BeliefUpdater
        from game.nash_solver import NashSolver
        from game.game_metrics import GameMetrics

        app.state.game           = StochasticGame(cfg.get("game", {}))
        app.state.attacker_model = AttackerModel(cfg.get("game", {}))
        app.state.belief_updater = BeliefUpdater()
        app.state.nash_solver    = NashSolver()
        app.state.game_metrics   = GameMetrics()
        logger.info("Game model initialised.")
    except Exception as exc:
        logger.warning("Game model failed: %s.", exc)
        app.state.game = None

    # ── Report generator ───────────────────────────────────────────────
    if app.state.llm_client:
        from explainability.report_generator import ReportGenerator
        app.state.report_generator = ReportGenerator(
            llm        = app.state.llm_client,
            auto_save  = True,
        )
    else:
        app.state.report_generator = None

    logger.info("All shared resources initialised.")


async def shutdown(app: Any) -> None:
    """Clean up resources on application shutdown."""
    if getattr(app.state, "kg_client", None):
        app.state.kg_client.close()
        logger.info("Neo4j client closed.")

    if getattr(app.state, "agent", None):
        try:
            app.state.agent.stop_training()
        except Exception:
            pass

    logger.info("Shutdown complete.")


# ══════════════════════════════════════════════════════════════════════════════
# Dependency providers (called by FastAPI Depends())
# ══════════════════════════════════════════════════════════════════════════════

def get_config(request: Request) -> dict:
    """Return the application config dict."""
    return getattr(request.app.state, "config", {})


def get_agent(request: Request) -> Optional[AgentType]:
    """Return the current ACD agent (may be None if not yet initialised)."""
    return getattr(request.app.state, "agent", None)


def require_agent(request: Request) -> AgentType:
    """
    Return the agent or raise 503 if not initialised.

    Use this for endpoints that strictly need the agent.
    """
    agent = getattr(request.app.state, "agent", None)
    if agent is None:
        raise HTTPException(
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE,
            detail      = (
                "Agent not initialised. "
                "POST /training/start to create and start the agent."
            ),
        )
    return agent


def get_env(request: Request) -> Optional[EnvType]:
    """Return the current CybORG environment."""
    return getattr(request.app.state, "env", None)


def get_kg_client(request: Request) -> Optional[Neo4jClientType]:
    """Return the Neo4j client (may be None if Neo4j is unavailable)."""
    return getattr(request.app.state, "kg_client", None)


def require_kg_client(request: Request) -> Neo4jClientType:
    """Return the KG client or raise 503."""
    client = getattr(request.app.state, "kg_client", None)
    if client is None:
        raise HTTPException(
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE,
            detail      = "Neo4j knowledge graph is not available.",
        )
    return client


def get_rag_retriever(request: Request) -> Optional[RAGRetrieverType]:
    """Return the RAG retriever."""
    return getattr(request.app.state, "rag_retriever", None)


def get_llm_client(request: Request) -> Optional[Any]:
    """Return the LLM client."""
    return getattr(request.app.state, "llm_client", None)


def get_drift_detector(request: Request) -> Optional[DriftDetectorType]:
    """Return the drift detector."""
    return getattr(request.app.state, "drift_detector", None)


def get_game(request: Request) -> Optional[GameType]:
    """Return the stochastic game model."""
    return getattr(request.app.state, "game", None)


def get_belief_updater(request: Request) -> Optional[Any]:
    """Return the Bayesian belief updater."""
    return getattr(request.app.state, "belief_updater", None)


def get_nash_solver(request: Request) -> Optional[Any]:
    """Return the Nash solver."""
    return getattr(request.app.state, "nash_solver", None)


def get_game_metrics(request: Request) -> Optional[Any]:
    """Return the game metrics tracker."""
    return getattr(request.app.state, "game_metrics", None)


def get_attacker_model(request: Request) -> Optional[Any]:
    """Return the attacker model."""
    return getattr(request.app.state, "attacker_model", None)


def get_ws_manager(request: Request) -> WebSocketMgrType:
    """Return the WebSocket connection manager."""
    return request.app.state.ws_manager


def get_alerts(request: Request) -> list:
    """Return the in-memory alert store."""
    return getattr(request.app.state, "alerts", [])


def get_report_generator(request: Request) -> Optional[ReportGenType]:
    """Return the report generator."""
    return getattr(request.app.state, "report_generator", None)