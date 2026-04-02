"""
api/routers/training.py
========================
Training lifecycle management endpoints.

Routes
------
    POST   /training/start      Start a new training run
    GET    /training/status     Get current training status + metrics
    DELETE /training/stop       Stop the running training
    POST   /training/pause      Pause training (save checkpoint)
    POST   /training/resume     Resume from checkpoint
    GET    /training/config     Get current training config
    PUT    /training/config     Update training hyperparameters
    GET    /training/history    Get training run history
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status

from api.dependencies import (
    get_agent, get_config, get_drift_detector,
    get_ws_manager, require_agent,
)
from api.schemas.training import (
    TrainingConfig, TrainingStartRequest,
    TrainingStatus, TrainingResult,
)

logger  = logging.getLogger(__name__)
router  = APIRouter()


# ══════════════════════════════════════════════════════════════════════════════
# Start training
# ══════════════════════════════════════════════════════════════════════════════

@router.post(
    "/start",
    response_model = TrainingStatus,
    status_code    = status.HTTP_202_ACCEPTED,
    summary        = "Start a new training run",
)
async def start_training(
    body:              TrainingStartRequest,
    background_tasks:  BackgroundTasks,
    request:           Request,
    config:            Dict = Depends(get_config),
):
    """
    Initialise the agent and environment, then start training asynchronously.

    Returns immediately with a 202 Accepted.  Poll ``GET /training/status``
    for progress updates, or subscribe to the ``ws/training`` WebSocket.
    """
    if request.app.state.agent and request.app.state.agent.is_training:
        raise HTTPException(
            status_code = status.HTTP_409_CONFLICT,
            detail      = "Training is already in progress. "
                          "POST /training/stop to stop it first.",
        )

    # Merge request overrides into base config
    train_cfg = {**config.get("agent", {}), **body.model_dump(exclude_none=True)}

    try:
        # Build env and agent
        from envs.env_factory import make_env
        from agents.registry import AgentRegistry
        from drift.detector_factory import DetectorFactory

        env   = make_env(train_cfg, n_envs=train_cfg.get("n_envs", 1))
        agent = AgentRegistry.build(env, train_cfg)

        request.app.state.env   = env
        request.app.state.agent = agent

        # Rebuild drift detector with new config
        drift_cfg = train_cfg.get("drift", config.get("drift", {}))
        request.app.state.drift_detector = DetectorFactory.build(drift_cfg)

    except Exception as exc:
        logger.error("Failed to initialise agent: %s", exc)
        raise HTTPException(
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail      = f"Agent initialisation failed: {exc}",
        )

    total_steps = body.total_timesteps or train_cfg.get("total_timesteps", 1_000_000)
    ws_manager  = getattr(request.app.state, "ws_manager", None)

    async def _train():
        try:
            agent.start_training()
            # Run training in thread pool (SB3 learn() is synchronous)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: agent.learn(total_timesteps=total_steps),
            )
        except asyncio.CancelledError:
            logger.info("Training task cancelled.")
        except Exception as exc:
            logger.error("Training error: %s", exc)
        finally:
            agent.stop_training()
            if ws_manager:
                await ws_manager.broadcast_json(
                    {"event": "training_complete", "metrics": agent.get_metrics()}
                )

    task = asyncio.create_task(_train())
    request.app.state.training_task = task

    logger.info("Training started — %d steps, agent=%s", total_steps, agent)
    return TrainingStatus(
        is_training     = True,
        total_timesteps = total_steps,
        agent_type      = train_cfg.get("agent_type", "cvar_ppo"),
        message         = "Training started.",
    )


# ══════════════════════════════════════════════════════════════════════════════
# Status
# ══════════════════════════════════════════════════════════════════════════════

@router.get(
    "/status",
    response_model = TrainingStatus,
    summary        = "Get training status and live metrics",
)
async def get_training_status(
    request: Request,
    agent    = Depends(get_agent),
):
    """Return current training status and live metrics."""
    if agent is None:
        return TrainingStatus(
            is_training = False,
            message     = "No agent loaded.",
        )

    metrics = agent.get_metrics()
    return TrainingStatus(
        is_training          = agent.is_training,
        total_timesteps      = metrics.get("total_timesteps", 0),
        episode_count        = metrics.get("episode_count", 0),
        mean_reward          = metrics.get("mean_reward"),
        cvar_005             = metrics.get("cvar_005"),
        loss_policy          = metrics.get("loss_policy"),
        loss_value           = metrics.get("loss_value"),
        loss_ewc             = metrics.get("loss_ewc"),
        training_elapsed_s   = metrics.get("training_elapsed_s", 0.0),
        agent_type           = metrics.get("agent_type", "unknown"),
        device               = metrics.get("device", "cpu"),
        message              = "Training in progress." if agent.is_training else "Training stopped.",
    )


# ══════════════════════════════════════════════════════════════════════════════
# Stop
# ══════════════════════════════════════════════════════════════════════════════

@router.delete(
    "/stop",
    summary = "Stop the running training run",
)
async def stop_training(
    request: Request,
    agent   = Depends(require_agent),
):
    """Stop the currently running training and save a checkpoint."""
    task = getattr(request.app.state, "training_task", None)
    if task and not task.done():
        task.cancel()

    agent.stop_training()
    path = agent.save_checkpoint(tag="manual_stop")

    return {
        "stopped":         True,
        "checkpoint_path": str(path),
        "timesteps":       agent.total_timesteps_trained,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/config", summary="Get current training config")
async def get_training_config(config: Dict = Depends(get_config)):
    """Return the current agent + training configuration."""
    return config.get("agent", {})


@router.put("/config", summary="Update training hyperparameters")
async def update_training_config(
    body:    Dict[str, Any],
    request: Request,
):
    """
    Update training hyperparameters.

    Changes take effect on the NEXT training run, not the current one.
    """
    if request.app.state.agent and request.app.state.agent.is_training:
        raise HTTPException(
            status_code = status.HTTP_409_CONFLICT,
            detail      = "Cannot update config during active training.",
        )
    current = request.app.state.config.get("agent", {})
    current.update(body)
    return {"updated": True, "config": current}


# ══════════════════════════════════════════════════════════════════════════════
# History
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/history", summary="Get training run history")
async def get_training_history(request: Request):
    """Return the list of completed training runs (from DB)."""
    try:
        from db.session import get_db_session
        # Placeholder — real impl queries TrainingRun model
        return {"runs": [], "message": "Database not connected."}
    except Exception:
        return {"runs": []}