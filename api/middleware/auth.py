"""
api/middleware/auth.py
=======================
JWT Bearer token validation for protected API endpoints.

The ACD API uses JWT (HS256) tokens issued by POST /auth/login.
Protected routes include everything except:
    /health, /health/*, /docs, /redoc, /openapi.json

Usage in route handlers
-----------------------
    from api.middleware.auth import require_auth
    from fastapi import Depends

    @router.get("/protected")
    async def protected(user = Depends(require_auth)):
        return {"user": user["sub"]}
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

logger = logging.getLogger(__name__)

# Public paths that do not require authentication
_PUBLIC_PATHS = {
    "/health", "/health/live", "/health/ready", "/health/version",
    "/docs", "/redoc", "/openapi.json",
    "/auth/login", "/auth/refresh",
    "/metrics",
}

_bearer_scheme = HTTPBearer(auto_error=False)


def require_auth(
    request:     Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
) -> Dict[str, Any]:
    """
    Dependency that validates the Bearer JWT token.

    Returns the decoded token payload dict on success.
    Raises 401 if the token is missing or invalid.

    Skip by setting ``ACD_AUTH_DISABLED=true`` in environment
    (useful for development / testing).
    """
    if os.getenv("ACD_AUTH_DISABLED", "false").lower() == "true":
        return {"sub": "dev_user", "role": "admin"}

    if request.url.path in _PUBLIC_PATHS:
        return {"sub": "anonymous"}

    if credentials is None:
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail      = "Missing Bearer token. Include Authorization: Bearer <token>.",
            headers     = {"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials
    payload = _verify_token(token)

    if payload is None:
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail      = "Invalid or expired token.",
            headers     = {"WWW-Authenticate": "Bearer"},
        )

    return payload


def require_admin(user: Dict[str, Any] = Depends(require_auth)) -> Dict[str, Any]:
    """Dependency that requires admin role."""
    if user.get("role") not in ("admin",):
        raise HTTPException(
            status_code = status.HTTP_403_FORBIDDEN,
            detail      = "Admin role required.",
        )
    return user


def _verify_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Verify and decode a JWT token.

    Returns decoded payload or None if invalid.
    """
    secret = os.getenv("ACD_JWT_SECRET", "acd-dev-secret-change-in-production")

    try:
        import jwt
        payload = jwt.decode(token, secret, algorithms=["HS256"])
        return payload
    except ImportError:
        logger.warning("PyJWT not installed — auth disabled. pip install PyJWT")
        return {"sub": "dev_user", "role": "admin"}
    except Exception as exc:
        logger.debug("Token verification failed: %s", exc)
        return None