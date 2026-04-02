"""
api/schemas/evaluation.py
==========================
Pydantic v2 models for the evaluation and benchmark endpoints.

Used by ``api/routers/evaluation.py``.

Models
------
    EvalResult     — Single evaluation run result (all paper metrics)
    BenchmarkRow   — One row of the benchmark comparison table
    BenchmarkTable — Full Table 1 from the paper
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class EvalResult(BaseModel):
    """Result of a single evaluation run — matches ``evaluate.run_eval()`` output."""

    n_episodes: int = 0
    mean_reward: Optional[float] = None
    std_reward: Optional[float] = None
    min_reward: Optional[float] = None
    max_reward: Optional[float] = None
    cvar_001: Optional[float] = Field(default=None, description="CVaR at α=0.01")
    cvar_005: Optional[float] = Field(default=None, description="CVaR at α=0.05")
    cvar_010: Optional[float] = Field(default=None, description="CVaR at α=0.10")
    cvar_020: Optional[float] = Field(default=None, description="CVaR at α=0.20")
    cvar_050: Optional[float] = Field(default=None, description="CVaR at α=0.50")
    success_rate: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Fraction of episodes with positive reward."
    )
    catastrophic_rate: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Fraction of episodes with reward < -10."
    )
    elapsed_s: Optional[float] = None

    model_config = {"extra": "allow"}


class BenchmarkRow(BaseModel):
    """One row in the benchmark comparison (Paper Table 1)."""

    agent: str = Field(description="Agent label, e.g. 'CVaR-PPO + EWC (Ours)'.")
    mean_reward: Optional[float] = None
    cvar_005: Optional[float] = None
    success_rate: Optional[float] = None
    catastrophic_rate: Optional[float] = None
    is_ours: bool = False
    error: Optional[str] = Field(default=None, description="Error message if agent failed.")


class BenchmarkTable(BaseModel):
    """Full benchmark comparison table — Paper Table 1."""

    rows: List[BenchmarkRow] = Field(default_factory=list)
