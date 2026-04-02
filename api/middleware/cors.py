"""
api/middleware/cors.py
=======================
CORS origin configuration for development and production.

Development  : Allow localhost:3000 (Next.js dev server)
Production   : Allow only the configured FRONTEND_URL
"""

from __future__ import annotations

import os
from typing import Dict, List


def get_cors_origins(config: Dict = None) -> List[str]:
    """
    Return the list of allowed CORS origins based on environment.

    Priority:
        1. CORS_ORIGINS env var (comma-separated)
        2. config["cors"]["origins"] list
        3. Default: localhost development URLs
    """
    env_origins = os.getenv("CORS_ORIGINS")
    if env_origins:
        return [o.strip() for o in env_origins.split(",")]

    if config:
        cfg_origins = config.get("cors", {}).get("origins")
        if cfg_origins:
            return cfg_origins

    # Default: dev origins
    return [
        "http://localhost:3000",    # Next.js dev server
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        "http://localhost:8080",    # Alternative dev port
    ]