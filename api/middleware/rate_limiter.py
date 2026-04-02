"""
api/middleware/rate_limiter.py
===============================
Rate limiting via slowapi (Starlette-compatible limiter).

Default limits:
    General endpoints      : 120/minute
    Training start/stop    : 10/minute
    LLM explanation        : 30/minute
    KG rebuild             : 2/hour

Rate limits are applied per client IP.
"""

from __future__ import annotations

import logging
from typing import Callable

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded
    from slowapi.util import get_remote_address
    _SLOWAPI_AVAILABLE = True
except ImportError:
    _SLOWAPI_AVAILABLE = False
    logger.debug("slowapi not installed — rate limiting disabled.")


# Global limiter instance (import this in route handlers)
limiter: "Limiter | None" = None


def register_rate_limiter(app: FastAPI) -> None:
    """Attach the rate limiter to the FastAPI app."""
    global limiter

    if not _SLOWAPI_AVAILABLE:
        logger.warning("slowapi not installed — rate limiting disabled. "
                       "Install with: pip install slowapi")
        return

    limiter = Limiter(key_func=get_remote_address, default_limits=["120/minute"])
    app.state.limiter = limiter

    # Register the 429 handler
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    logger.info("Rate limiter registered (120/min default).")


def get_limiter() -> "Limiter | None":
    """Return the global limiter instance."""
    return limiter