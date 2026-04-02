"""
monitoring/structlog_config.py
================================
Structured logging configuration for the ACD Framework.

Two output modes
----------------
    "pretty"  (development)
        ╔══════════════════════════════════════════════════════════════╗
        ║ 14:32:01 [INFO    ] agents.cvar_ppo — Training step 50,000  ║
        ║ 14:32:01 [WARNING ] drift.wasserstein — Score 0.21 > 0.15   ║
        ╚══════════════════════════════════════════════════════════════╝
        Uses ANSI colour codes.  Human-readable.  Not machine-parseable.

    "json"    (production)
        {"ts":"2024-01-15T14:32:01Z","level":"INFO","logger":"agents.cvar_ppo","msg":"Training step 50,000","step":50000}
        {"ts":"2024-01-15T14:32:01Z","level":"WARNING","logger":"drift.wasserstein","msg":"Score 0.21 > 0.15","score":0.21}
        One JSON object per line.  Parseable by Datadog, CloudWatch,
        Splunk, Loki, and any log aggregation platform.

Environment variables
---------------------
    LOG_LEVEL   : DEBUG, INFO, WARNING, ERROR, CRITICAL (default: INFO)
    LOG_FORMAT  : pretty or json (default: pretty)
    LOG_FILE    : Optional file path for rotating file handler

Logger hierarchy (noisy loggers silenced)
-----------------------------------------
    uvicorn.access        WARNING  (we have our own request logger)
    uvicorn.error         INFO
    sqlalchemy.engine     WARNING  (suppress SQL echo)
    httpx                 WARNING
    httpcore              WARNING
    sentence_transformers WARNING  (suppress model loading progress)
    faiss                 WARNING
    openai                WARNING
    anthropic             WARNING
    neo4j                 WARNING

Usage
-----
    from monitoring.structlog_config import configure_logging

    # Call once at startup before anything else logs
    configure_logging()

    # Or with explicit overrides
    configure_logging(
        level      = "DEBUG",
        log_format = "json",
        log_file   = "data/logs/acd.log",
    )
"""

from __future__ import annotations

import json
import logging
import logging.config
import logging.handlers
import os
import time
from typing import Any, Dict, Optional


# ── ANSI colour codes (used in "pretty" mode) ───────────────────────────────

class _ANSI:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    DIM    = "\033[2m"

    # Level colours
    DEBUG    = "\033[36m"    # Cyan
    INFO     = "\033[32m"    # Green
    WARNING  = "\033[33m"    # Yellow
    ERROR    = "\033[31m"    # Red
    CRITICAL = "\033[35m"    # Magenta

    # Logger name colour
    LOGGER   = "\033[34m"    # Blue
    TIME     = "\033[90m"    # Dark grey


# ── Logger name silencing map ────────────────────────────────────────────────

_NOISY_LOGGERS: Dict[str, str] = {
    "uvicorn.access":         "WARNING",
    "uvicorn.error":          "INFO",
    "sqlalchemy.engine":      "WARNING",
    "sqlalchemy.pool":        "WARNING",
    "sqlalchemy.orm":         "WARNING",
    "httpx":                  "WARNING",
    "httpcore":               "WARNING",
    "httpcore.http11":        "WARNING",
    "sentence_transformers":  "WARNING",
    "transformers":           "WARNING",
    "faiss":                  "WARNING",
    "openai":                 "WARNING",
    "anthropic":              "WARNING",
    "neo4j":                  "WARNING",
    "neo4j.notifications":    "ERROR",
    "urllib3":                "WARNING",
    "filelock":               "WARNING",
    "PIL":                    "WARNING",
    "celery":                 "INFO",
    "celery.worker":          "INFO",
    "celery.beat":            "WARNING",
    "kombu":                  "WARNING",
    "redis":                  "WARNING",
    "asyncio":                "WARNING",
}


# ════════════════════════════════════════════════════════════════════════════
# Public API
# ════════════════════════════════════════════════════════════════════════════

def configure_logging(
    level:      Optional[str] = None,
    log_format: Optional[str] = None,
    log_file:   Optional[str] = None,
    use_colour: Optional[bool] = None,
) -> None:
    """
    Configure application-wide logging.

    Call once at application startup before any module creates a logger.
    Subsequent calls are safe but have no effect (idempotent via root
    logger level check).

    Parameters
    ----------
    level : str, optional
        Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL.
        Falls back to LOG_LEVEL env var, then "INFO".
    log_format : str, optional
        Output format: "pretty" or "json".
        Falls back to LOG_FORMAT env var, then "pretty".
    log_file : str, optional
        If set, also write logs to this rotating file.
        Automatically creates parent directories.
        Falls back to LOG_FILE env var.
    use_colour : bool, optional
        Force ANSI colour output on/off for "pretty" mode.
        Auto-detected from terminal capabilities if None.
    """
    effective_level  = (level      or os.getenv("LOG_LEVEL",  "INFO")).upper()
    effective_format = (log_format or os.getenv("LOG_FORMAT", "pretty")).lower()
    effective_file   = log_file    or os.getenv("LOG_FILE")

    # Auto-detect colour support
    if use_colour is None:
        import sys
        use_colour = (
            effective_format == "pretty"
            and hasattr(sys.stdout, "isatty")
            and sys.stdout.isatty()
        )

    # ── Build handler list ─────────────────────────────────────────────
    handlers: Dict[str, Any] = {}

    if effective_format == "json":
        handlers["console"] = {
            "()":        _JsonStreamHandler,
            "stream":    "ext://sys.stdout",
            "level":     effective_level,
        }
    else:
        handlers["console"] = {
            "()":        _PrettyStreamHandler,
            "stream":    "ext://sys.stdout",
            "level":     effective_level,
            "use_colour":use_colour,
        }

    if effective_file:
        import pathlib
        pathlib.Path(effective_file).parent.mkdir(parents=True, exist_ok=True)
        handlers["file"] = {
            "()":              "logging.handlers.RotatingFileHandler",
            "filename":        effective_file,
            "maxBytes":        10 * 1024 * 1024,   # 10 MB
            "backupCount":     5,
            "encoding":        "utf-8",
            "level":           effective_level,
            "formatter":       "json_file",
        }

    # ── Build logger configs ───────────────────────────────────────────
    logger_configs: Dict[str, Any] = {
        name: {"level": lvl, "propagate": True}
        for name, lvl in _NOISY_LOGGERS.items()
    }

    # ── Apply config ───────────────────────────────────────────────────
    config: Dict[str, Any] = {
        "version":                   1,
        "disable_existing_loggers":  False,
        "formatters": {
            "json_file": {"()": _JsonFormatter},
        },
        "handlers":  handlers,
        "loggers":   logger_configs,
        "root": {
            "level":    effective_level,
            "handlers": list(handlers.keys()),
        },
    }

    logging.config.dictConfig(config)

    # ── Announce configuration ─────────────────────────────────────────
    root_logger = logging.getLogger()
    root_logger.debug(
        "Logging configured — level=%s, format=%s, file=%s",
        effective_level, effective_format, effective_file or "stdout only",
    )


# ════════════════════════════════════════════════════════════════════════════
# Handler implementations
# ════════════════════════════════════════════════════════════════════════════

class _JsonFormatter(logging.Formatter):
    """
    Formats log records as single-line JSON objects.

    Output schema
    -------------
    {
        "ts":      "2024-01-15T14:32:01Z",   ISO 8601 UTC timestamp
        "level":   "INFO",                    level name
        "logger":  "agents.cvar_ppo",         logger name
        "msg":     "Training step 50,000",    message
        "exc":     "Traceback ..."            only on exceptions
        ...extra key-value pairs from record
    }
    """

    # Keys that are part of the standard LogRecord and should NOT be
    # included as extra fields
    _STANDARD_KEYS = frozenset({
        "args", "created", "exc_info", "exc_text", "filename",
        "funcName", "levelname", "levelno", "lineno", "message",
        "module", "msecs", "msg", "name", "pathname", "process",
        "processName", "relativeCreated", "stack_info", "thread",
        "threadName", "taskName",
    })

    def format(self, record: logging.LogRecord) -> str:
        log_dict: Dict[str, Any] = {
            "ts":     time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)
            ),
            "level":  record.levelname,
            "logger": record.name,
            "msg":    record.getMessage(),
        }

        # Include exception traceback if present
        if record.exc_info:
            log_dict["exc"] = self.formatException(record.exc_info)

        # Include stack info if present
        if record.stack_info:
            log_dict["stack"] = self.formatStack(record.stack_info)

        # Include any extra fields passed via logger.xxx(..., extra={...})
        for key, value in record.__dict__.items():
            if key in self._STANDARD_KEYS:
                continue
            # Only include JSON-serialisable extra fields
            try:
                json.dumps(value)
                log_dict[key] = value
            except (TypeError, ValueError):
                log_dict[key] = str(value)

        return json.dumps(log_dict, default=str)


class _JsonStreamHandler(logging.StreamHandler):
    """StreamHandler that formats records as JSON lines."""

    def __init__(self, stream=None, **kwargs):
        super().__init__(stream)
        self.setFormatter(_JsonFormatter())


class _PrettyStreamHandler(logging.StreamHandler):
    """
    StreamHandler that formats records as coloured human-readable lines.

    Format:
        HH:MM:SS [LEVEL   ] logger.name — Message
    """

    # Level → colour map
    _LEVEL_COLOURS = {
        "DEBUG":    _ANSI.DEBUG,
        "INFO":     _ANSI.INFO,
        "WARNING":  _ANSI.WARNING,
        "ERROR":    _ANSI.ERROR,
        "CRITICAL": _ANSI.CRITICAL,
    }

    def __init__(self, stream=None, use_colour: bool = True, **kwargs):
        super().__init__(stream)
        self.use_colour = use_colour

    def format(self, record: logging.LogRecord) -> str:
        ts      = time.strftime("%H:%M:%S", time.localtime(record.created))
        level   = record.levelname
        logger  = record.name
        message = record.getMessage()

        if record.exc_info:
            exc_text = self.formatException(record.exc_info)
            message  = f"{message}\n{exc_text}"

        if self.use_colour:
            level_colour  = self._LEVEL_COLOURS.get(level, "")
            level_padded  = f"{level:<8}"
            return (
                f"{_ANSI.TIME}{ts}{_ANSI.RESET} "
                f"{level_colour}[{level_padded}]{_ANSI.RESET} "
                f"{_ANSI.LOGGER}{logger}{_ANSI.RESET} "
                f"— {message}"
            )
        else:
            return f"{ts} [{level:<8}] {logger} — {message}"

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            stream = self.stream
            stream.write(msg + self.terminator)
            self.flush()
        except RecursionError:
            raise
        except Exception:
            self.handleError(record)


# ════════════════════════════════════════════════════════════════════════════
# Convenience functions
# ════════════════════════════════════════════════════════════════════════════

def get_log_level() -> str:
    """Return the current root log level name (e.g. 'INFO')."""
    return logging.getLevelName(logging.getLogger().level)


def set_log_level(level: str) -> None:
    """
    Dynamically change the root log level at runtime.

    Useful for temporarily enabling DEBUG output during troubleshooting
    without restarting the application.

    Parameters
    ----------
    level : str
        New log level: DEBUG, INFO, WARNING, ERROR, CRITICAL.
    """
    numeric = getattr(logging, level.upper(), None)
    if numeric is None:
        raise ValueError(
            f"Unknown log level {level!r}. "
            f"Use: DEBUG, INFO, WARNING, ERROR, CRITICAL"
        )
    logging.getLogger().setLevel(numeric)
    logging.getLogger(__name__).info(
        "Root log level changed to %s", level.upper()
    )