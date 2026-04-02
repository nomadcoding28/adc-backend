"""api/routers/evaluation.py — Evaluation and benchmark endpoints."""

from typing import List, Optional
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request
from api.dependencies import get_agent, require_agent
from api.schemas.evaluation import EvalResult, BenchmarkTable

router = APIRouter()


@router.post("/run", status_code=202, summary="Start evaluation run")
async def run_evaluation(
    request          : Request,
    background_tasks : BackgroundTasks,
    n_episodes       : int = Query(50, ge=1, le=500),
    agent            = Depends(require_agent),
):
    """
    Run a full evaluation episode batch and compute all paper metrics.

    Results are stored and retrievable via GET /evaluation/latest.
    """
    async def _eval():
        try:
            from evaluate import run_eval
            result = run_eval(agent=agent, n_episodes=n_episodes)
            request.app.state.last_eval_result = result
        except Exception as exc:
            request.app.state.last_eval_result = {"error": str(exc)}

    background_tasks.add_task(_eval)
    return {"message": f"Evaluation started ({n_episodes} episodes).", "n_episodes": n_episodes}


@router.get("/latest", response_model=EvalResult, summary="Get latest evaluation result")
async def get_latest_result(request: Request):
    """Return the most recent evaluation run result."""
    result = getattr(request.app.state, "last_eval_result", None)
    if result is None:
        raise HTTPException(status_code=404, detail="No evaluation results yet.")
    return result


@router.get("/benchmark", response_model=BenchmarkTable, summary="Full benchmark table (paper Table 1)")
async def get_benchmark(request: Request):
    """
    Return the benchmark comparison table across all agent variants.

    This is Table 1 from the paper: CVaR-PPO+EWC vs ablations vs baselines.
    """
    result = getattr(request.app.state, "last_eval_result", None)
    if result is None:
        # Return placeholder
        return BenchmarkTable(rows=[])
    return result.get("benchmark_table", BenchmarkTable(rows=[]))


@router.get("/ablation", summary="Novelty ablation study results")
async def get_ablation(request: Request):
    """Return per-novelty contribution breakdown for the ablation table."""
    result = getattr(request.app.state, "last_eval_result", None)
    if result is None:
        return {"rows": []}
    return result.get("ablation", {"rows": []})