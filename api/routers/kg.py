"""api/routers/kg.py — Knowledge Graph query endpoints."""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from api.dependencies import get_kg_client, require_kg_client
from api.schemas.kg import KGGraph, AttackChain, CVENode

router = APIRouter()


@router.get("/graph", response_model=KGGraph, summary="Get full KG for D3 visualisation")
async def get_graph(
    client   = Depends(require_kg_client),
    limit    : int   = Query(200, ge=10, le=1000),
    min_cvss : float = Query(0.0, ge=0.0, le=10.0),
):
    """Return all nodes and edges for the D3 force-directed graph viewer."""
    from knowledge.kg_queries import KGQuerier
    querier = KGQuerier(client)
    return querier.get_full_graph(limit=limit, min_cvss=min_cvss)


@router.get("/stats", summary="Knowledge graph statistics")
async def get_kg_stats(client=Depends(require_kg_client)):
    """Return node/edge counts for the KG dashboard panel."""
    return client.get_stats()


@router.get("/cve/{cve_id}", response_model=CVENode, summary="Get CVE details")
async def get_cve(cve_id: str, client=Depends(require_kg_client)):
    """Return full details for a single CVE node."""
    from knowledge.kg_queries import KGQuerier
    querier = KGQuerier(client)
    result  = querier.get_cve(cve_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"CVE {cve_id!r} not found in KG.")
    return result


@router.get("/cve/{cve_id}/attack-chain", response_model=AttackChain, summary="Get attack chain for CVE")
async def get_attack_chain(cve_id: str, client=Depends(require_kg_client)):
    """Return the predicted ATT&CK kill chain for a given CVE."""
    from knowledge.kg_queries import KGQuerier
    return KGQuerier(client).get_attack_chain(cve_id)


@router.get("/technique/{technique_id}", summary="Get ATT&CK technique details")
async def get_technique(technique_id: str, client=Depends(require_kg_client)):
    """Return full details for a single ATT&CK technique node."""
    from knowledge.kg_queries import KGQuerier
    result = KGQuerier(client).get_technique(technique_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Technique {technique_id!r} not found.")
    return result


@router.get("/search", summary="Full-text search across CVEs and techniques")
async def search_kg(
    q        : str  = Query(..., min_length=2),
    node_type: Optional[str] = Query(None, description="CVE, Technique, or Tactic"),
    limit    : int  = Query(20, ge=1, le=100),
    client   = Depends(require_kg_client),
):
    """Full-text search across the knowledge graph."""
    from knowledge.kg_queries import KGQuerier
    return KGQuerier(client).search(query=q, node_type=node_type, limit=limit)


@router.post("/rebuild", summary="Trigger full KG rebuild (admin)")
async def rebuild_kg(client=Depends(require_kg_client)):
    """Enqueue a background KG rebuild task."""
    try:
        from tasks.kg_tasks import rebuild_kg_task
        task = rebuild_kg_task.delay()
        return {"task_id": task.id, "message": "KG rebuild enqueued."}
    except Exception:
        # Celery not available — run inline (slow)
        return {"message": "Celery not available. Configure it for async rebuilds."}