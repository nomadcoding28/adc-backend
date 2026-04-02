"""api/routers/cvar.py — CVaR risk metrics and distribution endpoints."""

from typing import List, Optional
from fastapi import APIRouter, Depends, Query
from api.dependencies import get_agent
from api.schemas.cvar import CVaRMetrics, AlphaSensitivity

router = APIRouter()


@router.get("/metrics", response_model=CVaRMetrics, summary="Get current CVaR metrics")
async def get_cvar_metrics(agent=Depends(get_agent)):
    """Return CVaR risk metrics from the most recent evaluation window."""
    if agent is None:
        return CVaRMetrics()
    metrics = agent.get_metrics()
    return CVaRMetrics(
        cvar_001         = metrics.get("cvar_001"),
        cvar_005         = metrics.get("cvar_005"),
        cvar_010         = metrics.get("cvar_010"),
        cvar_020         = metrics.get("cvar_020"),
        cvar_050         = metrics.get("cvar_050"),
        mean_reward      = metrics.get("mean_reward"),
        min_reward       = metrics.get("min_reward"),
        max_reward       = metrics.get("max_reward"),
        catastrophic_rate= metrics.get("catastrophic_rate"),
        tail_samples     = metrics.get("tail_samples"),
        alpha            = metrics.get("alpha", 0.05),
    )


@router.get("/distribution", summary="Get reward distribution histogram")
async def get_reward_distribution(
    agent   = Depends(get_agent),
    n_bins  : int = Query(20, ge=5, le=100),
):
    """Return histogram data for the reward distribution chart."""
    if agent is None:
        return {"bins": [], "counts": [], "var": None, "cvar": None}
    try:
        optimizer = agent.get_cvar_optimizer()
        return optimizer.get_distribution(n_bins=n_bins)
    except Exception:
        return {"bins": [], "counts": [], "var": None, "cvar": None}


@router.get("/alpha", response_model=List[AlphaSensitivity], summary="CVaR α-sensitivity table")
async def get_alpha_sensitivity(agent=Depends(get_agent)):
    """
    Return CVaR values for multiple α levels — the paper's Table 2.
    """
    if agent is None:
        return []
    alphas = [0.01, 0.05, 0.10, 0.20, 0.50]
    try:
        optimizer = agent.get_cvar_optimizer()
        return [
            AlphaSensitivity(
                alpha       = a,
                cvar_value  = optimizer.compute_cvar(alpha=a),
                var_value   = optimizer.compute_var(alpha=a),
                interpretation = _alpha_label(a),
            )
            for a in alphas
        ]
    except Exception:
        return [
            AlphaSensitivity(alpha=a, interpretation=_alpha_label(a))
            for a in alphas
        ]


@router.get("/ablation", summary="CVaR-PPO vs Standard PPO ablation")
async def get_ablation(agent=Depends(get_agent)):
    """Return ablation comparison table (CVaR-PPO vs baseline)."""
    if agent is None:
        return {"rows": []}
    metrics = agent.get_metrics()
    return {
        "rows": [
            {
                "agent":            "CVaR-PPO + EWC (Ours)",
                "mean_reward":      metrics.get("mean_reward"),
                "cvar_005":         metrics.get("cvar_005"),
                "success_rate":     metrics.get("success_rate"),
                "catastrophic_rate":metrics.get("catastrophic_rate"),
            }
        ]
    }


def _alpha_label(alpha: float) -> str:
    if alpha <= 0.01:  return "Extremely risk-averse (worst 1%)"
    if alpha <= 0.05:  return "Balanced — default setting (worst 5%)"
    if alpha <= 0.10:  return "Moderately risk-averse (worst 10%)"
    if alpha <= 0.20:  return "Mildly risk-averse (worst 20%)"
    return "Near risk-neutral (≈ mean)"