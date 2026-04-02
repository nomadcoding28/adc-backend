"""
api/schemas/drift.py
=====================
Pydantic v2 models for the concept drift detection endpoints.

Used by ``api/routers/drift.py``.

Models
------
    DriftScore   — Current Wasserstein drift score snapshot
    DriftEvent   — Single drift event record
    DriftHistory — Full drift score + event history
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DriftScore(BaseModel):
    """Current drift detection score snapshot."""

    score: float = Field(default=0.0, description="Current Wasserstein distance.")
    threshold: float = Field(default=0.15, description="Drift detection threshold.")
    is_drifting: bool = Field(default=False, description="Whether drift is currently detected.")
    n_events: int = Field(default=0, description="Total drift events detected so far.")
    step: int = Field(default=0, description="Current global step count.")


class DriftDimension(BaseModel):
    """Per-dimension Wasserstein distance for the most shifted dimensions."""

    dim: int = Field(description="Observation dimension index (0–53).")
    w1: float = Field(description="Wasserstein-1 distance for this dimension.")


class DriftEvent(BaseModel):
    """Single drift detection event record."""

    step: int = Field(description="Global step when drift was detected.")
    distance: float = Field(description="Wasserstein distance at detection.")
    threshold: float = Field(default=0.15)
    detector_type: str = Field(default="Wasserstein")
    top_shifted_dims: List[DriftDimension] = Field(default_factory=list)
    wasserstein_distance: Optional[float] = None
    use_pca: bool = False
    timestamp: Optional[str] = None

    model_config = {"extra": "allow"}


class DriftHistory(BaseModel):
    """Full drift score and event history for the dashboard."""

    scores: List[float] = Field(default_factory=list, description="All recorded distances.")
    events: List[DriftEvent] = Field(default_factory=list, description="Detected drift events.")
    threshold: float = Field(default=0.15)
