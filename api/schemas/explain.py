"""
api/schemas/explain.py
=======================
Pydantic v2 models for the LLM explainability endpoints.

Used by ``api/routers/explain.py``.

Models
------
    ExplanationRequest — POST /explain/action request body
    ExplanationCard    — Structured 4-part ReAct explanation response
    ReActStep          — Single ReAct trace step
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ══════════════════════════════════════════════════════════════════════════════
#  Request
# ══════════════════════════════════════════════════════════════════════════════

class ExplanationRequest(BaseModel):
    """Request body for POST /explain/action."""

    action: str = Field(description="Human-readable action description, e.g. 'Isolate Host-3'.")
    action_idx: Optional[int] = Field(default=None, description="Numeric action index (0–53).")
    step: int = Field(default=0, description="Current episode step.")
    threat: str = Field(default="", description="Brief threat description.")
    obs_decoded: Optional[Dict[str, Any]] = Field(
        default=None, description="Decoded observation from ObservationProcessor."
    )
    risk_score: float = Field(default=0.0, ge=0.0, le=1.0, description="CVaR risk score.")
    attacker_type: str = Field(default="Unknown", description="Predicted attacker type from belief.")
    technique_ids: List[str] = Field(
        default_factory=list, description="MITRE ATT&CK technique IDs, e.g. ['T1190']."
    )
    cve_ids: List[str] = Field(
        default_factory=list, description="CVE identifiers, e.g. ['CVE-2021-44228']."
    )
    action_success: bool = Field(default=True, description="Whether the action succeeded.")
    reward_breakdown: Optional[Dict[str, float]] = None

    model_config = {"extra": "allow"}


# ══════════════════════════════════════════════════════════════════════════════
#  Response
# ══════════════════════════════════════════════════════════════════════════════

class ReActStep(BaseModel):
    """Single step in the ReAct reasoning trace."""

    step_type: str = Field(description="OBSERVE, THINK, ACT, or RESULT.")
    output: str = Field(default="", description="LLM generated text for this step.")
    prompt_tokens: int = Field(default=0)
    latency_s: float = Field(default=0.0)


class ExplanationCard(BaseModel):
    """
    Structured 4-part explanation card produced by the ReAct agent.

    The four sections map to the ReAct loop:
        1. threat_detected  ← OBSERVE step output
        2. why_action       ← THINK + ACT step outputs
        3. risk_assessment  ← Derived from CVaR + belief
        4. mitre_mapping    ← ATT&CK technique / CVE references
    """

    card_id: Optional[str] = Field(default=None, description="Unique explanation ID.")
    action: str = Field(default="", description="Action that was explained.")
    step: int = Field(default=0)

    # Four-part explanation
    threat_detected: str = Field(default="", description="What is happening in the network.")
    why_action: str = Field(default="", description="Why the agent chose this action.")
    risk_assessment: str = Field(default="", description="Risk context from CVaR + game model.")
    mitre_mapping: str = Field(default="", description="Mapped ATT&CK techniques and CVEs.")

    # Trace metadata
    trace: List[ReActStep] = Field(default_factory=list)
    n_docs_retrieved: int = Field(default=0)
    total_tokens: int = Field(default=0)
    latency_s: float = Field(default=0.0)

    # Context
    attacker_type: str = Field(default="Unknown")
    risk_score: float = Field(default=0.0)
    technique_ids: List[str] = Field(default_factory=list)
    cve_ids: List[str] = Field(default_factory=list)
    action_success: bool = Field(default=True)

    model_config = {"extra": "allow"}
