"""
auth/password.py
=================
bcrypt password hashing and verification.

Usage
-----
    hashed = hash_password("my_password")
    ok     = verify_password("my_password", hashed)   # True
    ok     = verify_password("wrong",       hashed)   # False
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    import bcrypt
    _BCRYPT_AVAILABLE = True
except ImportError:
    _BCRYPT_AVAILABLE = False
    logger.warning("bcrypt not installed — passwords stored in plaintext (dev only). "
                   "Install with: pip install bcrypt")


def hash_password(plain: str) -> str:
    """
    Hash a plaintext password with bcrypt.

    Parameters
    ----------
    plain : str
        Plaintext password.

    Returns
    -------
    str
        bcrypt hash string (includes salt).
    """
    if not _BCRYPT_AVAILABLE:
        # Dev fallback — NEVER use in production
        return f"plaintext:{plain}"

    salt   = bcrypt.gensalt(rounds=12)
    hashed = bcrypt.hashpw(plain.encode("utf-8"), salt)
    return hashed.decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    """
    Verify a plaintext password against a bcrypt hash.

    Parameters
    ----------
    plain : str
        Plaintext password to check.
    hashed : str
        bcrypt hash (from ``hash_password``).

    Returns
    -------
    bool
        True if the password matches.
    """
    if not _BCRYPT_AVAILABLE:
        return hashed == f"plaintext:{plain}"

    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except Exception as exc:
        logger.warning("Password verification error: %s", exc)
        return False


def is_strong_password(password: str) -> tuple[bool, str]:
    """
    Check if a password meets minimum strength requirements.

    Returns
    -------
    tuple(is_strong: bool, reason: str)
    """
    if len(password) < 8:
        return False, "Password must be at least 8 characters."
    if not any(c.isdigit() for c in password):
        return False, "Password must contain at least one digit."
    if not any(c.isalpha() for c in password):
        return False, "Password must contain at least one letter."
    return True, "OK"