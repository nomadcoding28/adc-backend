"""api/routers/checkpoints.py — Model checkpoint management."""

from pathlib import Path
from typing import List
from fastapi import APIRouter, Depends, HTTPException, Request
from api.dependencies import require_agent
from pydantic import BaseModel

router = APIRouter()


class CheckpointLoadRequest(BaseModel):
    path: str


@router.get("/", summary="List available checkpoints")
async def list_checkpoints(agent=Depends(require_agent)):
    """Return all saved checkpoint files in the checkpoint directory."""
    ckpt_dir = agent.checkpoint_dir
    if not ckpt_dir.exists():
        return {"checkpoints": []}

    files = sorted(ckpt_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    return {
        "checkpoints": [
            {
                "name":         f.name,
                "path":         str(f),
                "size_mb":      round(f.stat().st_size / (1024**2), 2),
                "modified_at":  f.stat().st_mtime,
            }
            for f in files
        ]
    }


@router.post("/save", summary="Save current model checkpoint")
async def save_checkpoint(agent=Depends(require_agent)):
    """Save the current agent weights as a checkpoint."""
    path = agent.save_checkpoint(tag="manual")
    return {"saved": True, "path": str(path)}


@router.post("/load", summary="Load a checkpoint")
async def load_checkpoint(
    body    : CheckpointLoadRequest,
    request : Request,
):
    """Load an agent from a saved checkpoint file."""
    p = Path(body.path)
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"Checkpoint not found: {p}")

    try:
        from agents.registry import AgentRegistry
        env     = request.app.state.env
        config  = request.app.state.config.get("agent", {})
        config["agent_type"] = "cvar_ppo"

        agent = AgentRegistry.build(env, config)
        # SB3 load sets model weights from checkpoint
        agent.model = agent.model.__class__.load(str(p), env=env)
        request.app.state.agent = agent

        return {"loaded": True, "path": str(p)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load checkpoint: {exc}")


@router.delete("/{filename}", summary="Delete a checkpoint file")
async def delete_checkpoint(filename: str, agent=Depends(require_agent)):
    """Delete a checkpoint file by filename."""
    p = agent.checkpoint_dir / filename
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"Checkpoint {filename!r} not found.")
    p.unlink()
    return {"deleted": True, "filename": filename}