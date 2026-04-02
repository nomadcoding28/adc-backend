"""
api/middleware/error_handler.py
================================
Global exception handlers — converts every unhandled exception into a
clean, structured JSON error response.

All errors follow this schema:
    {
        "error":   "ShortErrorType",
        "detail":  "Human-readable message",
        "status":  400,
        "path":    "/api/route",
        "request_id": "abc123"
    }

No raw Python tracebacks ever reach the client in production.
"""

from __future__ import annotations

import logging
import traceback
import uuid

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


def register_exception_handlers(app: FastAPI) -> None:
    """Register all custom exception handlers on the FastAPI app."""

    # ── 422 Validation error ───────────────────────────────────────────
    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        errors = [
            {
                "field":   " → ".join(str(l) for l in e["loc"]),
                "message": e["msg"],
                "type":    e["type"],
            }
            for e in exc.errors()
        ]
        return JSONResponse(
            status_code = status.HTTP_422_UNPROCESSABLE_ENTITY,
            content     = {
                "error":   "ValidationError",
                "detail":  "Request body or parameters failed validation.",
                "errors":  errors,
                "status":  422,
                "path":    str(request.url.path),
            },
        )

    # ── HTTPException ──────────────────────────────────────────────────
    @app.exception_handler(HTTPException)
    async def http_exception_handler(
        request: Request, exc: HTTPException
    ) -> JSONResponse:
        return JSONResponse(
            status_code = exc.status_code,
            content     = {
                "error":  _status_label(exc.status_code),
                "detail": exc.detail,
                "status": exc.status_code,
                "path":   str(request.url.path),
            },
        )

    # ── Catch-all: 500 Internal Server Error ──────────────────────────
    @app.exception_handler(Exception)
    async def generic_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        request_id = str(uuid.uuid4())[:8]
        logger.error(
            "Unhandled exception [%s] %s %s: %s",
            request_id, request.method, request.url.path, exc,
            exc_info=True,
        )
        return JSONResponse(
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR,
            content     = {
                "error":      "InternalServerError",
                "detail":     "An unexpected error occurred. Check server logs.",
                "status":     500,
                "path":       str(request.url.path),
                "request_id": request_id,
            },
        )


def _status_label(code: int) -> str:
    labels = {
        400: "BadRequest",    401: "Unauthorized",
        403: "Forbidden",     404: "NotFound",
        405: "MethodNotAllowed", 409: "Conflict",
        422: "ValidationError", 429: "TooManyRequests",
        500: "InternalServerError", 503: "ServiceUnavailable",
    }
    return labels.get(code, f"HTTPError{code}")