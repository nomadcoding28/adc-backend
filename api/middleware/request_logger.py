"""
api/middleware/request_logger.py
=================================
Structured request/response logging middleware.

Logs every HTTP request and response with:
    - Method, path, status code, latency
    - Request ID (for log correlation)
    - Client IP

Uses structlog-compatible JSON output when structlog is installed,
otherwise falls back to stdlib logging.
"""

from __future__ import annotations

import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("api.access")

# Paths excluded from access logging (too noisy)
_SKIP_PATHS = {"/health", "/health/live", "/health/ready", "/metrics"}


class RequestLoggerMiddleware(BaseHTTPMiddleware):
    """Log all HTTP requests with method, path, status, and latency."""

    async def dispatch(self, request: Request, call_next) -> Response:
        if request.url.path in _SKIP_PATHS:
            return await call_next(request)

        request_id = str(uuid.uuid4())[:8]
        start      = time.monotonic()

        # Attach request_id to request state for use in handlers
        request.state.request_id = request_id

        try:
            response = await call_next(request)
        except Exception as exc:
            logger.error(
                "Request failed [%s] %s %s — %s",
                request_id, request.method, request.url.path, exc,
            )
            raise

        elapsed_ms = round((time.monotonic() - start) * 1000, 1)
        status     = response.status_code

        log_fn = logger.warning if status >= 400 else logger.info
        log_fn(
            "[%s] %s %s → %d  (%sms)  ip=%s",
            request_id,
            request.method,
            request.url.path,
            status,
            elapsed_ms,
            request.client.host if request.client else "unknown",
        )

        # Add request ID to response headers for client-side tracing
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{elapsed_ms}ms"

        return response