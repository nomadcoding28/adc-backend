"""
api/schemas/network.py
=======================
Pydantic v2 models for the network topology endpoints.

Used by ``api/routers/network.py``.

Models
------
    HostState     — State of a single CybORG network host
    TopologyGraph — Full network topology with host states
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class HostState(BaseModel):
    """State of a single host in the CybORG network."""

    name: str = Field(description="Host name, e.g. 'Enterprise0', 'Op_Server0'.")
    compromised: bool = Field(default=False)
    is_decoy: bool = Field(default=False)
    malicious_process: bool = Field(default=False)
    reachable: bool = Field(default=False)
    privileged_session: bool = Field(default=False)
    active_connections: int = Field(default=0, ge=0)

    model_config = {"extra": "allow"}


class TopologyGraph(BaseModel):
    """Full CybORG network topology snapshot for the dashboard."""

    hosts: List[HostState] = Field(default_factory=list)
    attacker_host_idx: int = Field(
        default=0,
        description="Index of the host with the most advanced attacker foothold.",
    )
    last_action_type: str = Field(
        default="Monitor",
        description="Most recent defender action type.",
    )
    last_action_success: bool = Field(default=True)
    step_fraction: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Current episode progress (step / max_steps).",
    )
    compromised_ratio: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Fraction of hosts currently compromised.",
    )

    model_config = {"extra": "allow"}
