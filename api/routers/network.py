"""api/routers/network.py — Network topology and host state endpoints."""

from fastapi import APIRouter, Depends, HTTPException, status
from api.dependencies import get_env, get_agent
from api.schemas.network import TopologyGraph, HostState

router = APIRouter()


@router.get("/topology", response_model=TopologyGraph, summary="Get live network topology")
async def get_topology(env=Depends(get_env)):
    """Return the current CybORG network topology with host statuses."""
    if env is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Environment not initialised.",
        )
    state = env.get_network_state()
    hosts = []
    for name, s in state.get("hosts", {}).items():
        hosts.append(HostState(
            name             = name,
            compromised      = s.get("compromised", False),
            is_decoy         = s.get("is_decoy", False),
            malicious_process= s.get("malicious_process", False),
            reachable        = s.get("reachable", False),
            privileged_session=s.get("privileged_session", False),
            active_connections=s.get("active_connections", 0),
        ))
    feedback = state.get("action_feedback", {})
    return TopologyGraph(
        hosts              = hosts,
        attacker_host_idx  = feedback.get("attacker_host_idx", 0),
        last_action_type   = feedback.get("last_action_type", "Monitor"),
        last_action_success= feedback.get("last_action_success", True),
        step_fraction      = feedback.get("step_fraction", 0.0),
        compromised_ratio  = feedback.get("compromised_ratio", 0.0),
    )


@router.get("/hosts", summary="Get all host states")
async def get_hosts(env=Depends(get_env)):
    """Return a flat list of all host states."""
    if env is None:
        return {"hosts": []}
    state = env.get_network_state()
    return {"hosts": state.get("hosts", {})}


@router.get("/hosts/{host_name}", response_model=HostState, summary="Get single host state")
async def get_host(host_name: str, env=Depends(get_env)):
    """Return the state of a single host."""
    if env is None:
        raise HTTPException(status_code=503, detail="Environment not initialised.")
    state = env.get_network_state()
    hosts = state.get("hosts", {})
    if host_name not in hosts:
        raise HTTPException(status_code=404, detail=f"Host {host_name!r} not found.")
    s = hosts[host_name]
    return HostState(name=host_name, **s)


@router.get("/metrics", summary="Get environment metrics")
async def get_env_metrics(env=Depends(get_env)):
    """Return current environment episode metrics."""
    if env is None:
        return {}
    return env.get_metrics()