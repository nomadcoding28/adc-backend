"""api/routers/incidents.py — Incident report endpoints."""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from api.dependencies import get_report_generator
from api.schemas.incidents import Incident, IncidentCreate

router = APIRouter()


@router.get("/", summary="List all incidents")
async def list_incidents(
    request  : Request,
    limit    : int           = Query(20, ge=1, le=100),
    offset   : int           = Query(0, ge=0),
    severity : Optional[str] = Query(None),
):
    """Return paginated incident list."""
    generator = request.app.state.report_generator
    if generator is None:
        return {"incidents": [], "total": 0}

    reports = generator.all_reports
    if severity:
        reports = [r for r in reports if r.severity == severity.upper()]

    return {
        "incidents": [r.to_dict() for r in reports[offset: offset + limit]],
        "total":     len(reports),
    }


@router.get("/{incident_id}", summary="Get incident report")
async def get_incident(incident_id: str, request: Request):
    """Return a full incident report by ID."""
    generator = request.app.state.report_generator
    if generator is None:
        raise HTTPException(status_code=503, detail="Report generator not available.")

    report = generator.get_report(incident_id)
    if report is None:
        raise HTTPException(status_code=404, detail=f"Incident {incident_id!r} not found.")
    return report.to_dict()


@router.get("/{incident_id}/markdown", summary="Get incident report as Markdown")
async def get_incident_markdown(incident_id: str, request: Request):
    """Return the raw Markdown content of an incident report."""
    generator = request.app.state.report_generator
    if generator is None:
        raise HTTPException(status_code=503, detail="Report generator not available.")
    report = generator.get_report(incident_id)
    if report is None:
        raise HTTPException(status_code=404, detail=f"Incident {incident_id!r} not found.")

    from fastapi.responses import PlainTextResponse
    return PlainTextResponse(report.markdown, media_type="text/markdown")


@router.post("/", status_code=201, summary="Create incident report")
async def create_incident(
    body      : IncidentCreate,
    generator = Depends(get_report_generator),
):
    """Trigger LLM generation of a new incident report."""
    if generator is None:
        raise HTTPException(status_code=503, detail="Report generator not available.")

    report = generator.generate(body.model_dump())
    return report.to_dict()


@router.get("/stats/summary", summary="Incident statistics")
async def incident_stats(request: Request):
    """Return incident count and severity breakdown."""
    generator = request.app.state.report_generator
    if generator is None:
        return {"total": 0}

    reports  = generator.all_reports
    by_sev   = {}
    by_type  = {}
    for r in reports:
        by_sev[r.severity]     = by_sev.get(r.severity, 0) + 1
        by_type[r.report_type] = by_type.get(r.report_type, 0) + 1

    return {
        "total":       len(reports),
        "by_severity": by_sev,
        "by_type":     by_type,
    }