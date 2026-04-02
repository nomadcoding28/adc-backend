"""
api/schemas/game.py
====================
Pydantic v2 models for the game-theoretic modelling endpoints.

Used by ``api/routers/game.py``.

Models
------
    BeliefState         — Bayesian posterior over attacker types
    AttackerPrediction  — Predicted attacker actions + recommendation
    GameState           — Current stochastic game state snapshot
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class BeliefState(BaseModel):
    """
    Bayesian belief over attacker types.

    The three types: ``Random``, ``APT``, ``Adaptive``.
    Probabilities sum to 1.0.
    """

    probabilities: Dict[str, float] = Field(
        default_factory=lambda: {"Random": 0.33, "APT": 0.34, "Adaptive": 0.33},
        description="Posterior probability per attacker type.",
    )
    dominant_type: Optional[str] = Field(
        default=None,
        description="Most likely attacker type (argmax of probabilities).",
    )
    dominant_probability: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Probability of the dominant type.",
    )
    n_updates: int = Field(default=0, description="Number of Bayesian updates applied.")
    last_update_step: Optional[int] = Field(
        default=None, description="Global step of the most recent belief update."
    )

    model_config = {"extra": "allow"}


class AttackerStrategy(BaseModel):
    """Action probabilities for a single attacker type."""

    type_name: str = Field(description="Attacker type label.")
    actions: Dict[str, float] = Field(
        default_factory=dict,
        description="Action → probability mapping for this attacker type.",
    )


class AttackerPrediction(BaseModel):
    """Predicted attacker next actions given current Bayesian belief."""

    dominant_type: Optional[str] = None
    dominant_probability: Optional[float] = None
    probabilities: Dict[str, float] = Field(default_factory=dict)
    strategies: Optional[List[AttackerStrategy]] = Field(
        default=None,
        description="Per-type action strategies for the current state.",
    )
    recommendation: Optional[str] = Field(
        default=None,
        description="Recommended defender strategy label.",
    )

    model_config = {"extra": "allow"}


class HostGameState(BaseModel):
    """State of a single host in the stochastic game."""

    name: str
    status: str = Field(default="clean", description="clean, compromised, isolated, or decoy.")
    is_target: bool = False


class GameState(BaseModel):
    """Current stochastic game state snapshot."""

    hosts: List[HostGameState] = Field(default_factory=list)
    n_compromised: int = Field(default=0)
    n_isolated: int = Field(default=0)
    n_clean: int = Field(default=0)
    step: int = Field(default=0)
    game_value: Optional[float] = Field(
        default=None, description="Current Nash equilibrium value V*."
    )
    breach_detected: bool = Field(default=False)

    model_config = {"extra": "allow"}
