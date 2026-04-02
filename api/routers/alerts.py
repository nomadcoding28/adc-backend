"""api/routers/alerts.py — Live alert feed endpoints and WebSocket."""

import time
from typing import List, Optional
from fastapi import APIRouter, Depends, Query, Request, WebSocket, WebSocketDisconnect
from api.dependencies import get_alerts, get_ws_manager
from api.schemas.alerts import Alert, AlertSeverity, AlertUpdate

router = APIRouter()


@router.get("/", response_model=List[Alert], summary="Get all alerts")
async def list_alerts(
    alerts    = Depends(get_alerts),
    severity  : Optional[str] = Query(None, description="Filter by severity"),
    limit     : int           = Query(50, ge=1, le=500),
    offset    : int           = Query(0, ge=0),
):
    """Return paginated alert list, optionally filtered by severity."""
    result = alerts
    if severity:
        result = [a for a in result if a.get("severity") == severity.upper()]
    return result[offset : offset + limit]


@router.get("/{alert_id}", response_model=Alert, summary="Get single alert")
async def get_alert(alert_id: str, alerts=Depends(get_alerts)):
    match = next((a for a in alerts if a.get("id") == alert_id), None)
    if not match:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Alert {alert_id!r} not found.")
    return match


@router.patch("/{alert_id}", summary="Update alert status (acknowledge)")
async def update_alert(alert_id: str, body: AlertUpdate, alerts=Depends(get_alerts)):
    for alert in alerts:
        if alert.get("id") == alert_id:
            alert.update(body.model_dump(exclude_none=True))
            return alert
    from fastapi import HTTPException
    raise HTTPException(status_code=404, detail=f"Alert {alert_id!r} not found.")


@router.get("/stats/summary", summary="Alert statistics")
async def alert_stats(alerts=Depends(get_alerts)):
    """Return counts by severity."""
    counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for a in alerts:
        sev = a.get("severity", "LOW")
        counts[sev] = counts.get(sev, 0) + 1
    return {"total": len(alerts), "by_severity": counts}


# ── WebSocket ────────────────────────────────────────────────────────────────

@router.websocket("/ws/alerts")
async def alerts_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for real-time alert streaming.

    Connect and receive alert JSON objects as they are generated.
    """
    ws_manager = websocket.app.state.ws_manager
    await ws_manager.connect(websocket, room="alerts")
    try:
        while True:
            # Keep connection alive — server pushes, client rarely sends
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket, room="alerts")