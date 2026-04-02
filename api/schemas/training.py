"""
api/schemas/training.py
========================
Pydantic v2 request / response models for the training lifecycle.

Used by ``api/routers/training.py``.

Models
------
    TrainingStartRequest — POST /training/start body
    TrainingConfig       — Hyper-parameter snapshot
    TrainingStatus       — GET  /training/status response
    TrainingResult       — Final training run result
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


# ══════════════════════════════════════════════════════════════════════════════
#  Request models
# ══════════════════════════════════════════════════════════════════════════════

class TrainingStartRequest(BaseModel):
    """Body for POST /training/start."""

    agent_type: Optional[str] = Field(
        default=None,
        description="Agent variant: 'cvar_ppo', 'standard_ppo', etc.",
    )
    total_timesteps: Optional[int] = Field(
        default=None,
        ge=1_000,
        le=50_000_000,
        description="Total environment steps to train.",
    )
    learning_rate: Optional[float] = Field(default=None, gt=0, le=1.0)
    n_steps: Optional[int] = Field(default=None, ge=64, le=16_384)
    batch_size: Optional[int] = Field(default=None, ge=8, le=4_096)
    n_epochs: Optional[int] = Field(default=None, ge=1, le=100)
    gamma: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    gae_lambda: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    clip_range: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    ent_coef: Optional[float] = Field(default=None, ge=0.0)
    vf_coef: Optional[float] = Field(default=None, ge=0.0)
    n_envs: Optional[int] = Field(default=None, ge=1, le=32)
    device: Optional[str] = Field(default=None, description="'cpu', 'cuda', or 'auto'.")

    # CVaR overrides
    cvar_alpha: Optional[float] = Field(default=None, gt=0.0, lt=1.0)
    cvar_weight: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    # EWC overrides
    ewc_lambda: Optional[float] = Field(default=None, ge=0.0)
    ewc_enabled: Optional[bool] = Field(default=None)

    # Resume
    resume_checkpoint: Optional[str] = Field(
        default=None,
        description="Path to checkpoint .zip to resume from.",
    )

    model_config = {"extra": "allow"}


class TrainingConfig(BaseModel):
    """Snapshot of the full training hyper-parameter set."""

    agent_type: str = "cvar_ppo"
    total_timesteps: int = 2_000_000
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    n_envs: int = 1
    device: str = "auto"

    # CVaR
    cvar_alpha: float = 0.05
    cvar_weight: float = 0.30

    # EWC
    ewc_enabled: bool = True
    ewc_lambda: float = 0.4

    model_config = {"extra": "allow"}


# ══════════════════════════════════════════════════════════════════════════════
#  Response models
# ══════════════════════════════════════════════════════════════════════════════

class TrainingStatus(BaseModel):
    """Live training status — returned by GET /training/status."""

    is_training: bool = False
    agent_type: Optional[str] = None
    total_timesteps: Optional[int] = None
    episode_count: Optional[int] = None
    mean_reward: Optional[float] = None
    cvar_005: Optional[float] = None
    loss_policy: Optional[float] = None
    loss_value: Optional[float] = None
    loss_ewc: Optional[float] = None
    training_elapsed_s: Optional[float] = None
    device: Optional[str] = None
    message: Optional[str] = None

    model_config = {"extra": "allow"}


class TrainingResult(BaseModel):
    """Final result of a completed training run."""

    status: str = "completed"
    run_id: Optional[str] = None
    agent_type: str = "cvar_ppo"
    total_timesteps: int = 0
    mean_reward: Optional[float] = None
    cvar_005: Optional[float] = None
    cvar_001: Optional[float] = None
    catastrophic_rate: Optional[float] = None
    ewc_forgetting: Optional[float] = None
    ewc_tasks: int = 0
    drift_events: int = 0
    elapsed_s: float = 0.0
    checkpoint_path: Optional[str] = None
    message: Optional[str] = None

    model_config = {"extra": "allow"}
