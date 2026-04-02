"""
api/schemas/incidents.py
=========================
Pydantic v2 models for the incident report endpoints.

Used by ``api/routers/incidents.py``.

Models
------
    IncidentCreate  — POST body to trigger LLM incident report generation
    Incident        — Full incident report record
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class IncidentCreate(BaseModel):
    """Request body for POST /incidents/ — trigger incident report generation."""

    title: Optional[str] = Field(
        default=None,
        description="Optional incident title; auto-generated if omitted.",
    )
    severity: str = Field(
        default="MEDIUM",
        description="Incident severity: CRITICAL, HIGH, MEDIUM, LOW.",
    )
    threat_description: str = Field(
        default="",
        description="Free-text description of the observed threat.",
    )
    affected_hosts: List[str] = Field(
        default_factory=list,
        description="List of affected host names.",
    )
    technique_ids: List[str] = Field(
        default_factory=list,
        description="MITRE ATT&CK technique IDs.",
    )
    cve_ids: List[str] = Field(
        default_factory=list,
        description="Related CVE identifiers.",
    )
    actions_taken: List[str] = Field(
        default_factory=list,
        description="Defender actions taken during the incident.",
    )
    obs_snapshot: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Decoded observation snapshot at time of incident.",
    )
    risk_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    model_config = {"extra": "allow"}


class Incident(BaseModel):
    """Full incident report record."""

    id: Optional[str] = Field(default=None, description="Unique incident identifier.")
    title: str = Field(default="")
    severity: str = Field(default="MEDIUM")
    report_type: str = Field(default="incident", description="Report type label.")
    timestamp: Optional[str] = Field(default=None, description="ISO 8601 creation timestamp.")

    # Content
    summary: str = Field(default="", description="Executive summary of the incident.")
    threat_analysis: str = Field(default="", description="Detailed threat analysis.")
    actions_taken: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    mitre_techniques: List[str] = Field(default_factory=list)
    cve_references: List[str] = Field(default_factory=list)
    affected_hosts: List[str] = Field(default_factory=list)

    # Metadata
    risk_score: Optional[float] = None
    markdown: Optional[str] = Field(
        default=None,
        description="Full Markdown content of the report.",
    )
    generation_latency_s: Optional[float] = None

    model_config = {"extra": "allow"}
