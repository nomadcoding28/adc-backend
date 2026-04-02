"""
api/schemas/alerts.py
======================
Pydantic v2 models for the alert system endpoints.

Used by ``api/routers/alerts.py``.

Models
------
    AlertSeverity — Severity level enum
    Alert         — Single alert record
    AlertUpdate   — PATCH body for alert acknowledgement
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class AlertSeverity(str, Enum):
    """Alert severity levels."""

    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


class Alert(BaseModel):
    """Single alert record."""

    id: Optional[str] = Field(default=None, description="Unique alert identifier.")
    severity: str = Field(default="LOW", description="CRITICAL, HIGH, MEDIUM, LOW, or INFO.")
    title: str = Field(default="", description="Short alert headline.")
    message: str = Field(default="", description="Detailed alert description.")
    source: str = Field(
        default="system",
        description="Alert source: 'drift', 'agent', 'system', 'game', etc.",
    )
    timestamp: Optional[str] = Field(default=None, description="ISO 8601 timestamp.")
    step: Optional[int] = Field(default=None, description="Global step when alert was raised.")
    acknowledged: bool = Field(default=False)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "allow"}


class AlertUpdate(BaseModel):
    """PATCH body for updating alert status (e.g. acknowledge)."""

    acknowledged: Optional[bool] = None
    severity: Optional[str] = None
    message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    model_config = {"extra": "allow"}
