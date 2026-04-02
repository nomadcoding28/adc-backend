"""api/routers/metrics.py — Prometheus metrics scrape endpoint."""

from fastapi import APIRouter, Depends, Request
from fastapi.responses import PlainTextResponse
from api.dependencies import get_agent, get_drift_detector

router = APIRouter()


@router.get(
    "/",
    response_class = PlainTextResponse,
    summary        = "Prometheus metrics scrape",
    include_in_schema= False,
)
async def prometheus_metrics(
    request  : Request,
    agent    = Depends(get_agent),
    detector = Depends(get_drift_detector),
):
    """
    Prometheus-format metrics endpoint.

    Scraped by Prometheus every 15s.  Exposes:
        acd_mean_reward, acd_cvar_005, acd_timesteps,
        acd_drift_events, acd_episodes, acd_ws_connections
    """
    lines = ["# ACD Framework Metrics"]

    def gauge(name, value, help_text=""):
        if value is None:
            return
        lines.append(f"# HELP {name} {help_text}")
        lines.append(f"# TYPE {name} gauge")
        lines.append(f"{name} {float(value):.6f}")

    if agent:
        m = agent.get_metrics()
        gauge("acd_mean_reward",   m.get("mean_reward",  0), "Mean episode reward")
        gauge("acd_cvar_005",      m.get("cvar_005",     0), "CVaR alpha=0.05")
        gauge("acd_timesteps",     m.get("total_timesteps", 0), "Total training steps")
        gauge("acd_episodes",      m.get("episode_count", 0), "Total episodes")
        gauge("acd_loss_policy",   m.get("loss_policy",   0), "Policy loss")
        gauge("acd_loss_ewc",      m.get("loss_ewc",      0), "EWC penalty loss")
        gauge("acd_is_training",   1 if agent.is_training else 0, "Training active flag")

    if detector:
        gauge("acd_drift_events",   detector.n_events,        "Total drift events")
        gauge("acd_drift_distance", detector.current_distance, "Current drift distance")

    ws_mgr = request.app.state.ws_manager
    gauge("acd_ws_connections", ws_mgr.n_connections, "Active WebSocket connections")

    return "\n".join(lines) + "\n"


@router.get("/json", summary="Metrics as JSON")
async def json_metrics(
    agent    = Depends(get_agent),
    detector = Depends(get_drift_detector),
    request  : Request = None,
):
    """Return all metrics as a JSON object (convenience endpoint)."""
    result = {}
    if agent:
        result["agent"] = agent.get_metrics()
    if detector:
        result["drift"] = detector.get_metrics()
    if request:
        result["websockets"] = request.app.state.ws_manager.n_connections
    return result