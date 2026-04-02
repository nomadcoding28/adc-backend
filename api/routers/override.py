"""api/routers/override.py — Human-in-the-loop override endpoints."""

import time
from fastapi import APIRouter, Depends, HTTPException, Request
from api.dependencies import get_env, get_ws_manager, require_agent
from pydantic import BaseModel

router = APIRouter()


class OverrideRequest(BaseModel):
    action_idx: int
    reason:     str = ""
    analyst_id: str = "anonymous"


@router.post("/action", summary="Execute human override action")
async def execute_override(
    body    : OverrideRequest,
    request : Request,
    agent   = Depends(require_agent),
    env     = Depends(get_env),
):
    """
    Execute a manual defender action, bypassing the AI agent.

    The override is logged as an incident event and broadcast via WebSocket.
    """
    if env is None:
        raise HTTPException(status_code=503, detail="Environment not initialised.")

    action_space = env.action_space
    if not (0 <= body.action_idx < action_space.n):
        raise HTTPException(
            status_code=422,
            detail=f"action_idx {body.action_idx} out of range [0, {action_space.n}).",
        )

    # Execute the action in the environment
    try:
        obs, reward, terminated, truncated, info = env.step(body.action_idx)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Action execution failed: {exc}")

    # Build override event
    override_event = {
        "type":       "human_override",
        "action_idx": body.action_idx,
        "reason":     body.reason,
        "analyst_id": body.analyst_id,
        "reward":     float(reward),
        "terminated": bool(terminated or truncated),
        "timestamp":  time.time(),
        "info":       info,
    }

    # Broadcast to WebSocket clients
    ws_manager = request.app.state.ws_manager
    await ws_manager.broadcast_json({"event": "override", **override_event})

    # Append to alert list
    alerts = request.app.state.alerts
    alerts.append({
        "id":       f"override-{int(time.time())}",
        "type":     "OVERRIDE",
        "severity": "HIGH",
        "title":    f"Human Override — Action {body.action_idx}",
        "desc":     body.reason,
        "timestamp":time.time(),
    })

    return override_event


@router.get("/actions", summary="List available override actions")
async def list_override_actions(env=Depends(get_env)):
    """Return all available action descriptions for the override dropdown."""
    if env is None:
        return {"actions": []}
    mapper = env._action_mapper
    return {
        "actions": [
            {"idx": i, "description": mapper.describe(i)}
            for i in range(mapper.n_actions)
        ]
    }