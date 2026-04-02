"""api/routers/drift.py — Concept drift detection endpoints."""

from typing import List, Optional
from fastapi import APIRouter, Depends, Query
from api.dependencies import get_drift_detector
from api.schemas.drift import DriftScore, DriftEvent, DriftHistory

router = APIRouter()


@router.get("/current", response_model=DriftScore, summary="Get current drift score")
async def get_current_drift(detector=Depends(get_drift_detector)):
    """Return the most recent Wasserstein drift score."""
    if detector is None:
        return DriftScore(score=0.0, threshold=0.15, is_drifting=False)
    return DriftScore(
        score      = detector.current_distance,
        threshold  = detector.detectors[0].threshold if detector.detectors else 0.15,
        is_drifting= detector.current_distance > (
            detector.detectors[0].threshold if detector.detectors else 0.15
        ),
        n_events   = detector.n_events,
        step       = detector._step_count,
    )


@router.get("/history", summary="Get full drift score history")
async def get_drift_history(
    detector = Depends(get_drift_detector),
    last_n   : Optional[int] = Query(None, description="Return last N scores"),
):
    """Return the full distance score history for the dashboard chart."""
    if detector is None:
        return {"scores": [], "events": []}
    scores = detector.distance_history
    if last_n:
        scores = scores[-last_n:]
    return {
        "scores":    scores,
        "events":    detector.drift_history,
        "threshold": detector.detectors[0].threshold if detector.detectors else 0.15,
    }


@router.get("/events", response_model=List[DriftEvent], summary="Get drift events")
async def get_drift_events(detector=Depends(get_drift_detector)):
    """Return all drift events detected so far."""
    if detector is None:
        return []
    return [DriftEvent(**e) for e in detector.drift_history]


@router.get("/stats", summary="Drift detection statistics")
async def get_drift_stats(detector=Depends(get_drift_detector)):
    """Return aggregated drift detection stats."""
    if detector is None:
        return {}
    return detector.get_metrics()


@router.post("/force-check", summary="Force immediate drift check")
async def force_drift_check(detector=Depends(get_drift_detector)):
    """Force a drift check outside the normal check frequency."""
    if detector is None:
        return {"drift_detected": False, "message": "Detector not available."}
    result = detector.force_check()
    return result.to_dict()