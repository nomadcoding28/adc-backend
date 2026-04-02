"""api/routers/explain.py — LLM explanation endpoints."""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from api.dependencies import get_rag_retriever, get_llm_client
from api.schemas.explain import ExplanationRequest, ExplanationCard

router = APIRouter()


@router.post("/action", response_model=ExplanationCard, summary="Explain a defender action")
async def explain_action(
    body    : ExplanationRequest,
    request : Request,
    retriever = Depends(get_rag_retriever),
    llm       = Depends(get_llm_client),
):
    """
    Generate a structured LLM explanation for a defender action.

    Runs the full ReAct (Observe → Think → Act → Result) pipeline and
    returns a structured ExplanationCard.
    """
    if retriever is None or llm is None:
        raise HTTPException(
            status_code=503,
            detail="Explainability pipeline not available (LLM or RAG not configured).",
        )

    try:
        from explainability.explanation_builder import ExplanationBuilder
        from explainability.react_agent import ReActAgent

        builder = ExplanationBuilder(llm=llm, retriever=retriever)
        agent   = ReActAgent(llm=llm, retriever=retriever, builder=builder)
        card    = agent.explain(body.model_dump())
        return card.to_dict()

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Explanation failed: {exc}")


@router.get("/history", summary="Get recent explanation history")
async def get_explanation_history(
    request : Request,
    last_n  : int = Query(20, ge=1, le=100),
):
    """Return the last N generated explanation cards."""
    # Stored on app state by the ReAct agent
    history = getattr(request.app.state, "explanation_history", [])
    return {"explanations": history[-last_n:], "total": len(history)}


@router.get("/history/{card_id}", summary="Get single explanation card")
async def get_explanation(card_id: str, request: Request):
    """Return a specific explanation card by ID."""
    history = getattr(request.app.state, "explanation_history", [])
    match   = next((c for c in history if c.get("card_id") == card_id), None)
    if not match:
        raise HTTPException(status_code=404, detail=f"Explanation {card_id!r} not found.")
    return match


@router.get("/rag/stats", summary="RAG pipeline statistics")
async def get_rag_stats(retriever=Depends(get_rag_retriever)):
    """Return RAG index statistics."""
    if retriever is None:
        return {"available": False}
    return {
        "available":  True,
        "n_indexed":  retriever.n_indexed,
        "is_ready":   retriever.is_ready,
        "top_k":      retriever.top_k,
    }