"""
api/schemas/kg.py
==================
Pydantic v2 models for the Knowledge Graph endpoints.

Used by ``api/routers/kg.py``.

Models
------
    KGNode      — Generic node in the knowledge graph
    KGEdge      — Relationship between two nodes
    KGGraph     — Full graph for D3 force-directed visualisation
    CVENode     — CVE vulnerability record
    TechniqueNode — MITRE ATT&CK technique record
    AttackStep  — One step in a predicted attack chain
    AttackChain — Full attack chain for a CVE
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class KGNode(BaseModel):
    """Generic node in the knowledge graph."""

    id: str = Field(description="Unique node identifier.")
    label: str = Field(default="", description="Display label for the node.")
    node_type: str = Field(default="generic", description="CVE, Technique, Tactic, Host, etc.")
    properties: Dict[str, Any] = Field(default_factory=dict)


class KGEdge(BaseModel):
    """Relationship between two KG nodes."""

    source: str = Field(description="Source node ID.")
    target: str = Field(description="Target node ID.")
    relationship: str = Field(default="RELATED_TO", description="Edge type / label.")
    weight: Optional[float] = None


class KGGraph(BaseModel):
    """Full knowledge graph for D3 force-directed visualisation."""

    nodes: List[KGNode] = Field(default_factory=list)
    edges: List[KGEdge] = Field(default_factory=list)
    n_nodes: int = Field(default=0)
    n_edges: int = Field(default=0)

    model_config = {"extra": "allow"}


class CVENode(BaseModel):
    """CVE vulnerability record from the knowledge graph."""

    cve_id: str = Field(description="CVE identifier, e.g. 'CVE-2021-44228'.")
    description: str = Field(default="")
    cvss_score: Optional[float] = Field(default=None, ge=0.0, le=10.0)
    cvss_vector: Optional[str] = None
    severity: Optional[str] = Field(default=None, description="LOW, MEDIUM, HIGH, CRITICAL.")
    cwe_ids: List[str] = Field(default_factory=list)
    affected_products: List[str] = Field(default_factory=list)
    techniques: List[str] = Field(
        default_factory=list,
        description="Linked ATT&CK technique IDs.",
    )
    published_date: Optional[str] = None
    last_modified: Optional[str] = None

    model_config = {"extra": "allow"}


class TechniqueNode(BaseModel):
    """MITRE ATT&CK technique record."""

    technique_id: str = Field(description="Technique ID, e.g. 'T1190'.")
    name: str = Field(default="")
    description: str = Field(default="")
    tactic: Optional[str] = Field(default=None, description="Parent tactic name.")
    platforms: List[str] = Field(default_factory=list)
    data_sources: List[str] = Field(default_factory=list)
    linked_cves: List[str] = Field(default_factory=list)


class AttackStep(BaseModel):
    """One step in a predicted attack chain."""

    order: int = Field(description="Step order in the chain (1-indexed).")
    technique_id: str
    technique_name: str = Field(default="")
    tactic: str = Field(default="")
    description: str = Field(default="")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class AttackChain(BaseModel):
    """Predicted ATT&CK kill chain for a given CVE."""

    cve_id: str
    steps: List[AttackStep] = Field(default_factory=list)
    total_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Product of individual step confidences.",
    )
