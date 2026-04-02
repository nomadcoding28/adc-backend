"""
api/app.py
==========
FastAPI application factory for the ACD Framework.

Creates the app instance, attaches middleware, mounts all routers,
registers the lifespan context manager (startup / shutdown), and
configures OpenAPI documentation.

Usage
-----
    # Development
    uvicorn api.app:app --reload --host 0.0.0.0 --port 8000

    # Production
    gunicorn api.app:app -k uvicorn.workers.UvicornWorker -w 4

    # Programmatic
    from api.app import create_app
    app = create_app(config)
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

# ── Router imports ──────────────────────────────────────────────────────────
from api.routers import (
    training, evaluation, network, alerts, cvar,
    drift, kg, explain, game, incidents,
    override, checkpoints, metrics, health,
)
from api.middleware.error_handler  import register_exception_handlers
from api.middleware.request_logger import RequestLoggerMiddleware
from api.middleware.rate_limiter   import register_rate_limiter
from api.middleware.cors           import get_cors_origins


# ══════════════════════════════════════════════════════════════════════════════
# Lifespan — startup and shutdown
# ══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan handler.

    On startup:
        - Initialise the shared application state (agent, env, KG client, etc.)
        - Build the RAG index
        - Start Celery workers (if configured)

    On shutdown:
        - Close DB connections
        - Close Neo4j driver
        - Flush logs
    """
    logger.info("ACD Framework API starting up...")

    # Import here to avoid circular imports
    from api.dependencies import startup, shutdown

    await startup(app)

    yield   # Application is running

    logger.info("ACD Framework API shutting down...")
    await shutdown(app)


# ══════════════════════════════════════════════════════════════════════════════
# App factory
# ══════════════════════════════════════════════════════════════════════════════

def create_app(config: Optional[Dict[str, Any]] = None) -> FastAPI:
    """
    Create and configure the FastAPI application.

    Parameters
    ----------
    config : dict, optional
        Application configuration dict.  If None, loads from config.yaml.

    Returns
    -------
    FastAPI
    """
    cfg = config or _load_default_config()

    app = FastAPI(
        title       = "ACD Framework API",
        description = (
            "Autonomous Cyber Defence Framework — REST API for training, "
            "monitoring, and controlling the ACD reinforcement learning system. "
            "Covers CVaR-PPO training, EWC continual learning, drift detection, "
            "knowledge graph, LLM explainability, and game-theoretic modelling."
        ),
        version         = "1.0.0",
        docs_url        = "/docs",
        redoc_url       = "/redoc",
        openapi_url     = "/openapi.json",
        lifespan        = lifespan,
        contact         = {
            "name":  "ACD Framework",
            "email": "acd@msrit.edu",
        },
        license_info    = {
            "name": "MIT",
        },
    )

    # Store config on app state for access in dependencies
    app.state.config = cfg

    # ── Middleware ─────────────────────────────────────────────────────
    _register_middleware(app, cfg)

    # ── Exception handlers ─────────────────────────────────────────────
    register_exception_handlers(app)

    # ── Rate limiter ───────────────────────────────────────────────────
    register_rate_limiter(app)

    # ── Routers ────────────────────────────────────────────────────────
    _mount_routers(app)

    logger.info("FastAPI app created with %d routes.", len(app.routes))
    return app


def _register_middleware(app: FastAPI, cfg: Dict[str, Any]) -> None:
    """Register all middleware in the correct order (outermost last)."""

    # Request / response logger (outermost)
    app.add_middleware(RequestLoggerMiddleware)

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins     = get_cors_origins(cfg),
        allow_credentials = True,
        allow_methods     = ["*"],
        allow_headers     = ["*"],
    )


def _mount_routers(app: FastAPI) -> None:
    """Mount all API routers with their prefixes and tags."""

    # Monitoring
    app.include_router(health.router,      prefix="/health",     tags=["Health"])
    app.include_router(network.router,     prefix="/network",    tags=["Network"])
    app.include_router(alerts.router,      prefix="/alerts",     tags=["Alerts"])

    # Agent
    app.include_router(training.router,    prefix="/training",   tags=["Training"])
    app.include_router(evaluation.router,  prefix="/evaluation", tags=["Evaluation"])
    app.include_router(cvar.router,        prefix="/cvar",       tags=["CVaR"])
    app.include_router(drift.router,       prefix="/drift",      tags=["Drift"])

    # Intelligence
    app.include_router(kg.router,          prefix="/kg",         tags=["Knowledge Graph"])
    app.include_router(explain.router,     prefix="/explain",    tags=["Explainability"])
    app.include_router(game.router,        prefix="/game",       tags=["Game Model"])

    # Reports
    app.include_router(incidents.router,   prefix="/incidents",  tags=["Incidents"])
    app.include_router(override.router,    prefix="/override",   tags=["Override"])
    app.include_router(checkpoints.router, prefix="/checkpoints",tags=["Checkpoints"])

    # Observability
    app.include_router(metrics.router,     prefix="/metrics",    tags=["Metrics"])


def _load_default_config() -> Dict[str, Any]:
    """Load configuration from config.yaml."""
    try:
        from utils.config_loader import load_config
        return load_config("config.yaml")
    except Exception as exc:
        logger.warning("Could not load config.yaml: %s — using defaults.", exc)
        return {}


# ── Module-level app instance (for uvicorn) ─────────────────────────────────
app = create_app()