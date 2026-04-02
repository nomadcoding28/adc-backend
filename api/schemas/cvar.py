"""
api/schemas/cvar.py
====================
Pydantic v2 models for the CVaR risk metrics endpoints.

Used by ``api/routers/cvar.py``.

Models
------
    CVaRMetrics      — Current CVaR risk metrics snapshot
    AlphaSensitivity — One row of α-sensitivity table (Paper Table 2)
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class CVaRMetrics(BaseModel):
    """CVaR risk metrics from the most recent evaluation or training window."""

    cvar_001: Optional[float] = Field(default=None, description="CVaR at α=0.01 (worst 1%).")
    cvar_005: Optional[float] = Field(default=None, description="CVaR at α=0.05 (worst 5%).")
    cvar_010: Optional[float] = Field(default=None, description="CVaR at α=0.10 (worst 10%).")
    cvar_020: Optional[float] = Field(default=None, description="CVaR at α=0.20 (worst 20%).")
    cvar_050: Optional[float] = Field(default=None, description="CVaR at α=0.50 (worst 50%).")
    mean_reward: Optional[float] = None
    min_reward: Optional[float] = None
    max_reward: Optional[float] = None
    catastrophic_rate: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Rate of episodes with reward < -10 (Op_Server0 breach).",
    )
    tail_samples: Optional[int] = Field(
        default=None,
        description="Number of samples in the CVaR tail window.",
    )
    alpha: float = Field(default=0.05, gt=0.0, lt=1.0, description="Active α risk level.")

    model_config = {"extra": "allow"}


class AlphaSensitivity(BaseModel):
    """One row of the α-sensitivity table — Paper Table 2."""

    alpha: float = Field(description="Risk level α ∈ (0, 1).")
    cvar_value: Optional[float] = Field(default=None, description="CVaR at this α.")
    var_value: Optional[float] = Field(default=None, description="VaR at this α.")
    interpretation: str = Field(
        default="",
        description="Human-readable risk label, e.g. 'Worst 5% — default setting'.",
    )
