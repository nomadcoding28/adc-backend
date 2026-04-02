"""
auth/guards.py
==============
Role-based access control FastAPI dependency decorators.

Usage
-----
    from auth.guards import role_required, admin_only

    @router.delete("/checkpoints/{id}")
    async def delete_ckpt(user = Depends(admin_only)):
        ...

    @router.post("/training/start")
    async def start_training(user = Depends(role_required("admin", "analyst"))):
        ...
"""

from __future__ import annotations

from typing import Callable, Dict, Any, Tuple

from fastapi import Depends, HTTPException, status
from api.middleware.auth import require_auth


def role_required(*roles: str) -> Callable:
    """
    FastAPI dependency factory — allows users with any of the given roles.

    Parameters
    ----------
    *roles : str
        Accepted role strings, e.g. ``role_required("admin", "analyst")``.
    """
    def _guard(user: Dict[str, Any] = Depends(require_auth)) -> Dict[str, Any]:
        if user.get("role") not in roles:
            raise HTTPException(
                status_code = status.HTTP_403_FORBIDDEN,
                detail      = (
                    f"Access denied. Required role: {' or '.join(roles)}. "
                    f"Your role: {user.get('role', 'none')}."
                ),
            )
        return user
    return _guard


def admin_only(user: Dict[str, Any] = Depends(require_auth)) -> Dict[str, Any]:
    """Dependency: allow admins only."""
    if user.get("role") != "admin":
        raise HTTPException(
            status_code = status.HTTP_403_FORBIDDEN,
            detail      = "Admin access required.",
        )
    return user


def analyst_or_above(user: Dict[str, Any] = Depends(require_auth)) -> Dict[str, Any]:
    """Dependency: allow admin or analyst roles."""
    if user.get("role") not in ("admin", "analyst"):
        raise HTTPException(
            status_code = status.HTTP_403_FORBIDDEN,
            detail      = "Analyst or admin access required.",
        )
    return user 