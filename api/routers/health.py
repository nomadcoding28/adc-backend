"""api/routers/health.py — Health, readiness, and liveness probes."""

from fastapi import APIRouter, Depends, Request
from api.dependencies import get_kg_client, get_drift_detector

router = APIRouter()


@router.get("/", summary="Full health check")
async def health(
    request:  Request,
    kg_client = Depends(get_kg_client),
):
    """Return health status of all system components."""
    return {
        "status":     "ok",
        "api":        "running",
        "neo4j":      "ok" if (kg_client and kg_client.ping()) else "unavailable",
        "agent":      "loaded" if request.app.state.agent else "not_loaded",
        "rag":        "loaded" if request.app.state.rag_retriever else "unavailable",
        "llm":        "loaded" if request.app.state.llm_client else "unavailable",
        "drift":      "active" if request.app.state.drift_detector else "unavailable",
        "game_model": "active" if request.app.state.game else "unavailable",
        "websockets": request.app.state.ws_manager.n_connections,
    }


@router.get("/ready", summary="Kubernetes readiness probe")
async def ready(request: Request):
    """Returns 200 when the API is ready to serve traffic."""
    return {"ready": True}


@router.get("/live", summary="Kubernetes liveness probe")
async def live():
    """Returns 200 as long as the process is alive."""
    return {"alive": True}


@router.get("/version", summary="API version info")
async def version():
    return {
        "version":   "1.0.0",
        "framework": "ACD Framework",
        "paper":     "CVaR-PPO + EWC Continual Learning for Autonomous Cyber Defence",
    }