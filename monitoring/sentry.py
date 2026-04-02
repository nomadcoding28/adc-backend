"""
monitoring/sentry.py
=====================
Sentry error tracking and performance monitoring for the ACD Framework.

Captures:
    - Unhandled exceptions from FastAPI route handlers
    - Unhandled exceptions from Celery background tasks
    - SQLAlchemy query errors
    - Manual captures via ``capture_exception()`` and ``capture_message()``

Integrations enabled automatically (when sentry-sdk is installed):
    - FastAPI     : Tags transactions with route name and method
    - Celery      : Tags task name, queue, and retry count
    - SQLAlchemy  : Breadcrumbs for DB queries in error traces
    - Logging     : Captures ERROR+ log records as Sentry events

Configuration
-------------
Set SENTRY_DSN environment variable to enable.  If unset, Sentry is
silently disabled — all calls become no-ops with no performance cost.

    SENTRY_DSN=https://xxx@o123.ingest.sentry.io/456
    export SENTRY_DSN=...

    # Or in .env file:
    SENTRY_DSN=https://xxx@o123.ingest.sentry.io/456

Performance monitoring
----------------------
``traces_sample_rate=0.1`` means 10% of requests are traced for
performance. Adjust in production based on your Sentry plan quota.

Usage
-----
    # Initialise once at startup (main.py)
    from monitoring.sentry import init_sentry
    init_sentry()

    # Manual exception capture with extra context
    from monitoring.sentry import capture_exception
    try:
        agent.learn(...)
    except Exception as exc:
        capture_exception(exc, extra={
            "step":       agent.total_timesteps_trained,
            "agent_type": "cvar_ppo",
            "cvar_005":   metrics.get("cvar_005"),
        })
        raise

    # Custom message (e.g. notable non-error events)
    from monitoring.sentry import capture_message
    capture_message(
        f"Drift detected at step {step} (W={distance:.3f})",
        level="warning",
        extra={"distance": distance, "threshold": threshold},
    )

    # Annotate code as a Sentry transaction span
    from monitoring.sentry import sentry_span
    with sentry_span("kg_rebuild", description="Full KG rebuild pipeline"):
        builder.build_full()
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional

logger = logging.getLogger(__name__)

# ── State ────────────────────────────────────────────────────────────────────
_SENTRY_INITIALISED: bool = False
_SENTRY_DSN:         Optional[str] = None


# ── Optional import ──────────────────────────────────────────────────────────
try:
    import sentry_sdk as _sdk
    from sentry_sdk.integrations.fastapi     import FastApiIntegration
    from sentry_sdk.integrations.starlette   import StarletteIntegration
    from sentry_sdk.integrations.celery      import CeleryIntegration
    from sentry_sdk.integrations.sqlalchemy  import SqlalchemyIntegration
    from sentry_sdk.integrations.logging     import LoggingIntegration
    _SENTRY_AVAILABLE = True
except ImportError:
    _sdk               = None
    FastApiIntegration = None
    _SENTRY_AVAILABLE  = False
    logger.debug(
        "sentry-sdk not installed — error tracking disabled. "
        "Install with: pip install sentry-sdk[fastapi,celery]"
    )


# ════════════════════════════════════════════════════════════════════════════
# Initialisation
# ════════════════════════════════════════════════════════════════════════════

def init_sentry(
    dsn:                Optional[str] = None,
    environment:        str           = "production",
    release:            str           = "acd-framework@1.0.0",
    traces_sample_rate: float         = 0.10,
    profiles_sample_rate: float       = 0.05,
    debug:              bool          = False,
) -> bool:
    """
    Initialise the Sentry SDK.

    Safe to call multiple times — only initialises on the first call.
    No-op if SENTRY_DSN is not set and dsn parameter is not provided.

    Parameters
    ----------
    dsn : str, optional
        Sentry DSN string.  Falls back to ``SENTRY_DSN`` env var.
    environment : str
        Environment tag shown in Sentry UI.
        Typically ``"production"``, ``"staging"``, or ``"development"``.
    release : str
        Release version string attached to every event.
        Convention: ``"service@major.minor.patch"``.
    traces_sample_rate : float
        Fraction of requests to trace for performance monitoring (0.0–1.0).
        Default 0.10 (10%).  Set 0 to disable performance monitoring.
    profiles_sample_rate : float
        Fraction of traced transactions to profile (0.0–1.0).
        Default 0.05 (5%).
    debug : bool
        Enable Sentry SDK debug logging.  Only use in development.

    Returns
    -------
    bool
        True if Sentry was successfully initialised.
    """
    global _SENTRY_INITIALISED, _SENTRY_DSN

    if _SENTRY_INITIALISED:
        logger.debug("Sentry already initialised — skipping.")
        return True

    effective_dsn = dsn or os.getenv("SENTRY_DSN")
    if not effective_dsn:
        logger.info(
            "SENTRY_DSN not set — error tracking disabled. "
            "Set SENTRY_DSN environment variable to enable."
        )
        return False

    if not _SENTRY_AVAILABLE:
        logger.warning(
            "sentry-sdk not installed — error tracking disabled. "
            "Install with: pip install 'sentry-sdk[fastapi,celery]'"
        )
        return False

    try:
        _sdk.init(
            dsn                   = effective_dsn,
            environment           = environment,
            release               = release,
            traces_sample_rate    = traces_sample_rate,
            profiles_sample_rate  = profiles_sample_rate,
            debug                 = debug,

            # ── Integrations ──────────────────────────────────────────
            integrations = [
                # FastAPI + Starlette: tag requests with route name
                StarletteIntegration(transaction_style="endpoint"),
                FastApiIntegration(transaction_style="endpoint"),

                # Celery: tag tasks with name, queue, retry count
                CeleryIntegration(
                    monitor_beat_tasks    = False,
                    propagate_traces      = True,
                ),

                # SQLAlchemy: add DB query breadcrumbs to error traces
                SqlalchemyIntegration(),

                # Logging: capture ERROR+ log records as Sentry events
                LoggingIntegration(
                    level       = logging.ERROR,
                    event_level = logging.ERROR,
                ),
            ],

            # ── Event filtering ───────────────────────────────────────
            before_send        = _before_send_filter,
            before_send_transaction = _before_send_transaction_filter,

            # ── Sensitive data scrubbing ──────────────────────────────
            # Automatically strip common PII field names
            send_default_pii   = False,

            # ── Breadcrumb limits ─────────────────────────────────────
            max_breadcrumbs    = 50,

            # ── Additional tags attached to every event ───────────────
            default_integrations = True,
        )

        # Set global tags visible in all Sentry events
        _sdk.set_tag("framework",  "acd-framework")
        _sdk.set_tag("component",  "backend")

        _SENTRY_INITIALISED = True
        _SENTRY_DSN         = effective_dsn

        logger.info(
            "Sentry initialised — environment=%r, release=%r, "
            "traces_rate=%.0f%%",
            environment, release, traces_sample_rate * 100,
        )
        return True

    except Exception as exc:
        logger.warning("Sentry initialisation failed: %s", exc)
        return False


# ════════════════════════════════════════════════════════════════════════════
# Event capture helpers
# ════════════════════════════════════════════════════════════════════════════

def capture_exception(
    exc:         Exception,
    extra:       Optional[Dict[str, Any]] = None,
    tags:        Optional[Dict[str, str]] = None,
    user:        Optional[Dict[str, Any]] = None,
    fingerprint: Optional[list]           = None,
) -> Optional[str]:
    """
    Capture and send an exception to Sentry.

    Also logs the exception locally at ERROR level, so it appears in
    application logs even when Sentry is disabled.

    Parameters
    ----------
    exc : Exception
        The exception to capture.
    extra : dict, optional
        Additional context to attach (e.g. {"step": 4821, "cvar": -2.14}).
    tags : dict, optional
        Searchable string tags in Sentry (e.g. {"agent_type": "cvar_ppo"}).
    user : dict, optional
        User information: {"id": "123", "username": "admin"}.
    fingerprint : list, optional
        Custom grouping fingerprint (overrides Sentry's default grouping).

    Returns
    -------
    str or None
        Sentry event ID if sent, None otherwise.
    """
    logger.error("Exception captured: %s", exc, exc_info=True)

    if not _SENTRY_INITIALISED or not _SENTRY_AVAILABLE:
        return None

    try:
        with _sdk.push_scope() as scope:
            if extra:
                for k, v in extra.items():
                    scope.set_extra(k, v)
            if tags:
                for k, v in tags.items():
                    scope.set_tag(k, v)
            if user:
                scope.set_user(user)
            if fingerprint:
                scope.fingerprint = fingerprint

            event_id = _sdk.capture_exception(exc)
            return event_id

    except Exception as capture_exc:
        logger.debug("Sentry capture_exception failed: %s", capture_exc)
        return None


def capture_message(
    message:     str,
    level:       str                      = "info",
    extra:       Optional[Dict[str, Any]] = None,
    tags:        Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """
    Send a custom message to Sentry.

    Use for notable non-error events: drift detected, KG rebuilt,
    training completed, etc.

    Parameters
    ----------
    message : str
        Human-readable message string.
    level : str
        Severity level: ``"debug"``, ``"info"``, ``"warning"``,
        ``"error"``, ``"fatal"``.
    extra : dict, optional
        Additional context.
    tags : dict, optional
        Searchable string tags.

    Returns
    -------
    str or None
        Sentry event ID if sent, None otherwise.
    """
    if not _SENTRY_INITIALISED or not _SENTRY_AVAILABLE:
        return None

    try:
        with _sdk.push_scope() as scope:
            if extra:
                for k, v in extra.items():
                    scope.set_extra(k, v)
            if tags:
                for k, v in tags.items():
                    scope.set_tag(k, v)

            event_id = _sdk.capture_message(message, level=level)
            return event_id

    except Exception as exc:
        logger.debug("Sentry capture_message failed: %s", exc)
        return None


@contextmanager
def sentry_span(
    op:          str,
    description: str = "",
    tags:        Optional[Dict[str, str]] = None,
) -> Generator[Any, None, None]:
    """
    Context manager that creates a Sentry performance span.

    Use this to trace expensive operations (KG rebuild, BERT encoding,
    Nash LP solve) within a larger transaction.

    Parameters
    ----------
    op : str
        Operation category (e.g. "kg.rebuild", "llm.explain", "drift.check").
    description : str
        Human-readable description shown in Sentry performance tab.
    tags : dict, optional
        Tags to attach to the span.

    Example
    -------
        with sentry_span("bert.encode", "Encoding 500 CVE descriptions"):
            embedder.embed_batch(cve_texts)
    """
    if not _SENTRY_INITIALISED or not _SENTRY_AVAILABLE:
        yield None
        return

    try:
        with _sdk.start_span(op=op, description=description) as span:
            if tags and span is not None:
                for k, v in tags.items():
                    span.set_tag(k, v)
            yield span
    except Exception:
        yield None


def set_user_context(user_id: str, username: str, role: str = "analyst") -> None:
    """
    Set the current user context for subsequent Sentry events.

    Call this after successful authentication.

    Parameters
    ----------
    user_id : str
    username : str
    role : str
    """
    if not _SENTRY_INITIALISED or not _SENTRY_AVAILABLE:
        return
    try:
        _sdk.set_user({"id": user_id, "username": username, "role": role})
    except Exception:
        pass


def clear_user_context() -> None:
    """Clear the current user context (call on logout)."""
    if not _SENTRY_INITIALISED or not _SENTRY_AVAILABLE:
        return
    try:
        _sdk.set_user(None)
    except Exception:
        pass


def add_breadcrumb(
    message:  str,
    category: str = "acd",
    level:    str = "info",
    data:     Optional[Dict[str, Any]] = None,
) -> None:
    """
    Add a breadcrumb to the current Sentry scope.

    Breadcrumbs appear in the trail of events leading up to an error,
    providing context about what the system was doing.

    Parameters
    ----------
    message : str
        Breadcrumb description.
    category : str
        Category string (e.g. "training", "drift", "kg").
    level : str
        ``"debug"``, ``"info"``, ``"warning"``, or ``"error"``.
    data : dict, optional
        Structured data attached to the breadcrumb.
    """
    if not _SENTRY_INITIALISED or not _SENTRY_AVAILABLE:
        return
    try:
        _sdk.add_breadcrumb(
            message  = message,
            category = category,
            level    = level,
            data     = data or {},
        )
    except Exception:
        pass


# ════════════════════════════════════════════════════════════════════════════
# Event filtering hooks
# ════════════════════════════════════════════════════════════════════════════

def _before_send_filter(
    event: Dict[str, Any],
    hint:  Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Pre-send filter — drop noisy / expected exceptions that clutter Sentry.

    Called by the SDK before every event is sent.
    Return None to drop the event; return event to send it.
    """
    exc_info = hint.get("exc_info")
    if exc_info and exc_info[0] is not None:
        exc_type = exc_info[0]

        # Drop expected, non-actionable exceptions
        _IGNORED_TYPES = (
            KeyboardInterrupt,
            SystemExit,
            BrokenPipeError,   # Client disconnected during streaming
        )
        if issubclass(exc_type, _IGNORED_TYPES):
            return None

        # Drop 4xx HTTP exceptions — these are user errors, not our bugs
        # Check for HTTPException from fastapi/starlette
        exc_name = exc_type.__name__
        if exc_name in ("HTTPException", "RequestValidationError"):
            exc_value = exc_info[1]
            status_code = getattr(exc_value, "status_code", 500)
            if isinstance(status_code, int) and status_code < 500:
                return None

    # Scrub any accidental password / token leaks in extra data
    if "extra" in event:
        event["extra"] = _scrub_sensitive(event["extra"])

    return event


def _before_send_transaction_filter(
    event: Dict[str, Any],
    hint:  Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Pre-send filter for performance transactions.

    Drop health check and metrics endpoints from performance traces —
    they are too frequent and not interesting for performance analysis.
    """
    transaction = event.get("transaction", "")
    _SKIP_TRANSACTIONS = {
        "/health", "/health/live", "/health/ready",
        "/metrics", "/metrics/json",
    }
    if transaction in _SKIP_TRANSACTIONS:
        return None
    return event


def _scrub_sensitive(data: Any) -> Any:
    """
    Recursively remove sensitive keys from event extra data.

    Keys matching any pattern in _SENSITIVE_KEYS are replaced with
    "[Filtered]" before the event is sent to Sentry.
    """
    _SENSITIVE_KEYS = {
        "password", "passwd", "secret", "token", "api_key",
        "apikey", "authorization", "cookie", "session", "dsn",
        "private_key", "jwt",
    }

    if isinstance(data, dict):
        return {
            k: "[Filtered]" if k.lower() in _SENSITIVE_KEYS else _scrub_sensitive(v)
            for k, v in data.items()
        }
    if isinstance(data, (list, tuple)):
        return type(data)(_scrub_sensitive(i) for i in data)
    return data


# ════════════════════════════════════════════════════════════════════════════
# Status helpers
# ════════════════════════════════════════════════════════════════════════════

def is_initialised() -> bool:
    """Return True if Sentry has been successfully initialised."""
    return _SENTRY_INITIALISED


def get_dsn_preview() -> str:
    """Return a redacted DSN preview safe for logging (hides the secret)."""
    if not _SENTRY_DSN:
        return "(not set)"
    parts = _SENTRY_DSN.split("@")
    if len(parts) == 2:
        return f"https://[key]@{parts[1]}"
    return "(configured)"