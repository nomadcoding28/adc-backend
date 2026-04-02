"""
main.py
=======
ACD Framework — application entry point.

Usage
-----
    # Development (hot reload)
    python main.py

    # Production (via uvicorn directly)
    uvicorn api.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Production (via gunicorn + uvicorn workers)
    gunicorn api.app:app -k uvicorn.workers.UvicornWorker -w 4 -b 0.0.0.0:8000
"""

from __future__ import annotations

import os
import sys

# Add project root to PYTHONPATH so all imports resolve correctly
sys.path.insert(0, os.path.dirname(__file__))


def main() -> None:
    """Start the ACD Framework API server."""
    from monitoring.structlog_config import configure_logging
    configure_logging()

    from monitoring.sentry import init_sentry
    init_sentry()

    from utils.seed import set_seed
    set_seed(int(os.getenv("ACD_SEED", "42")))

    import uvicorn
    uvicorn.run(
        "api.app:app",
        host        = os.getenv("HOST",    "0.0.0.0"),
        port        = int(os.getenv("PORT", "8000")),
        reload      = os.getenv("DEBUG", "false").lower() == "true",
        log_level   = os.getenv("LOG_LEVEL", "info").lower(),
        workers     = int(os.getenv("WORKERS", "1")),
        access_log  = False,    # We use our own request logger middleware
    )


if __name__ == "__main__":
    main()