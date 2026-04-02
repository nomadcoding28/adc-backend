"""
auth/router.py
==============
Authentication routes: login, refresh, logout.

POST /auth/login      → {access_token, refresh_token, token_type}
POST /auth/refresh    → {access_token, token_type}
POST /auth/logout     → {message}
GET  /auth/me         → current user info
"""

from __future__ import annotations

import time
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel

from auth.jwt_handler import sign_token, verify_token, create_refresh_token, decode_token_unsafe
from auth.password import verify_password

logger = router = APIRouter()
router = APIRouter()
logger = logging.getLogger(__name__)

# ── Default dev user (bypasses DB when ACD_AUTH_DISABLED=true) ──────────────
_DEV_USER = {
    "username":      "admin",
    "password_hash": "plaintext:admin",
    "role":          "admin",
    "email":         "admin@acd.local",
}


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token:  str
    refresh_token: str
    token_type:    str = "bearer"
    expires_in:    int = 28800


class RefreshRequest(BaseModel):
    refresh_token: str


@router.post("/login", response_model=TokenResponse, summary="Login and get JWT tokens")
async def login(body: LoginRequest):
    """
    Authenticate with username + password.
    Returns access_token (8h) and refresh_token (7d).
    """
    import os
    auth_disabled = os.getenv("ACD_AUTH_DISABLED", "false").lower() == "true"

    if auth_disabled:
        # Dev mode — accept any credentials
        sub  = body.username
        role = "admin"
    else:
        # Look up user (simplified — real impl queries DB)
        if body.username != _DEV_USER["username"]:
            raise HTTPException(
                status_code = status.HTTP_401_UNAUTHORIZED,
                detail      = "Invalid username or password.",
            )
        ok = verify_password(body.password, _DEV_USER["password_hash"])
        if not ok:
            raise HTTPException(
                status_code = status.HTTP_401_UNAUTHORIZED,
                detail      = "Invalid username or password.",
            )
        sub  = body.username
        role = _DEV_USER["role"]

    access_token  = sign_token({"sub": sub, "role": role, "type": "access"})
    refresh_token = create_refresh_token(sub=sub, role=role)

    logger.info("User %r logged in.", sub)
    return TokenResponse(
        access_token  = access_token,
        refresh_token = refresh_token,
    )


@router.post("/refresh", response_model=TokenResponse, summary="Refresh access token")
async def refresh(body: RefreshRequest):
    """Exchange a valid refresh token for a new access token."""
    payload = verify_token(body.refresh_token)
    if not payload or payload.get("type") != "refresh":
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail      = "Invalid or expired refresh token.",
        )

    sub  = payload["sub"]
    role = payload.get("role", "analyst")

    new_access  = sign_token({"sub": sub, "role": role})
    new_refresh = create_refresh_token(sub=sub, role=role)

    return TokenResponse(access_token=new_access, refresh_token=new_refresh)


@router.post("/logout", summary="Logout (client-side token invalidation)")
async def logout():
    """
    Logout endpoint.  Since tokens are stateless, the client discards them.
    A production system would blacklist the token in Redis.
    """
    return {"message": "Logged out. Discard your tokens on the client side."}


@router.get("/me", summary="Get current user info")
async def me(request: Request):
    """Return info about the currently authenticated user."""
    from api.middleware.auth import require_auth
    from fastapi import Depends
    # Extract token manually since we can't use Depends here easily
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated.")

    token   = auth_header[7:]
    payload = verify_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid token.")

    return {
        "sub":        payload.get("sub"),
        "role":       payload.get("role"),
        "token_type": "access",
    }