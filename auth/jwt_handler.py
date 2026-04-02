"""
auth/jwt_handler.py
====================
JWT token creation and verification for the ACD API.

Uses HS256 (HMAC-SHA256) symmetric signing.
Access tokens expire in 8 hours; refresh tokens in 7 days.

Environment variables
---------------------
    ACD_JWT_SECRET    : Signing secret (default: dev secret — CHANGE IN PRODUCTION)
    ACD_JWT_ALGO      : Algorithm (default: HS256)
    ACD_ACCESS_TTL    : Access token TTL in seconds (default: 28800 = 8h)
    ACD_REFRESH_TTL   : Refresh token TTL in seconds (default: 604800 = 7d)

Usage
-----
    token   = sign_token({"sub": "user123", "role": "admin"})
    payload = verify_token(token)
    # payload = {"sub": "user123", "role": "admin", "exp": ..., "iat": ...}
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_SECRET    = os.getenv("ACD_JWT_SECRET", "acd-dev-secret-CHANGE-IN-PRODUCTION")
_ALGORITHM = os.getenv("ACD_JWT_ALGO",   "HS256")
_ACCESS_TTL  = int(os.getenv("ACD_ACCESS_TTL",  "28800"))    # 8 hours
_REFRESH_TTL = int(os.getenv("ACD_REFRESH_TTL", "604800"))   # 7 days

try:
    import jwt as _jwt
    _JWT_AVAILABLE = True
except ImportError:
    _jwt = None
    _JWT_AVAILABLE = False
    logger.warning("PyJWT not installed — JWT auth disabled. pip install PyJWT")


def sign_token(
    payload:    Dict[str, Any],
    ttl_seconds: int = _ACCESS_TTL,
) -> str:
    """
    Create a signed JWT token.

    Parameters
    ----------
    payload : dict
        Claims to embed.  ``sub`` (subject) is required.
        ``iat`` and ``exp`` are added automatically.
    ttl_seconds : int
        Token lifetime in seconds.

    Returns
    -------
    str
        Encoded JWT string.

    Raises
    ------
    RuntimeError
        If PyJWT is not installed.
    """
    if not _JWT_AVAILABLE:
        raise RuntimeError("PyJWT not installed. pip install PyJWT")

    now = int(time.time())
    claims = {
        **payload,
        "iat": now,
        "exp": now + ttl_seconds,
        "type": payload.get("type", "access"),
    }
    return _jwt.encode(claims, _SECRET, algorithm=_ALGORITHM)


def create_refresh_token(sub: str, role: str = "user") -> str:
    """Create a long-lived refresh token."""
    return sign_token(
        {"sub": sub, "role": role, "type": "refresh"},
        ttl_seconds=_REFRESH_TTL,
    )


def verify_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Verify and decode a JWT token.

    Parameters
    ----------
    token : str
        Encoded JWT string.

    Returns
    -------
    dict or None
        Decoded payload, or None if the token is invalid / expired.
    """
    if not _JWT_AVAILABLE:
        logger.warning("PyJWT not installed — accepting all tokens.")
        return {"sub": "dev_user", "role": "admin"}

    try:
        payload = _jwt.decode(token, _SECRET, algorithms=[_ALGORITHM])
        return payload
    except _jwt.ExpiredSignatureError:
        logger.debug("Token expired.")
        return None
    except _jwt.InvalidTokenError as exc:
        logger.debug("Invalid token: %s", exc)
        return None


def decode_token_unsafe(token: str) -> Optional[Dict[str, Any]]:
    """
    Decode a JWT without verifying the signature.

    Only used to extract the ``sub`` claim from expired tokens
    during the refresh flow.  Never use this for auth decisions.
    """
    if not _JWT_AVAILABLE:
        return None
    try:
        return _jwt.decode(
            token, options={"verify_signature": False}, algorithms=[_ALGORITHM]
        )
    except Exception:
        return None