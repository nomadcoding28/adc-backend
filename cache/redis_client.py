"""
cache/redis_client.py
======================
Async Redis connection pool for the ACD Framework.

Used for:
    - Caching API responses (network topology, KG stats)
    - Celery task broker (when Celery is configured)
    - WebSocket session persistence
    - Rate limiter backend

Configuration
-------------
    REDIS_URL : redis://localhost:6379/0  (default)
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

_REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

try:
    import redis.asyncio as aioredis
    _REDIS_AVAILABLE = True
except ImportError:
    aioredis = None
    _REDIS_AVAILABLE = False
    logger.warning("redis[asyncio] not installed — caching disabled. pip install redis")


class RedisClient:
    """
    Async Redis client wrapper with JSON serialisation helpers.

    Parameters
    ----------
    url : str
        Redis connection URL.
    """

    def __init__(self, url: str = _REDIS_URL) -> None:
        self.url = url
        self._client = None

        if _REDIS_AVAILABLE:
            try:
                self._client = aioredis.from_url(
                    url,
                    encoding         = "utf-8",
                    decode_responses = True,
                    max_connections  = 20,
                )
                logger.info("Redis client created — %s", url.split("@")[-1])
            except Exception as exc:
                logger.warning("Redis connection failed: %s — caching disabled.", exc)

    async def get(self, key: str) -> Optional[Any]:
        """Get a JSON value by key. Returns None on miss or error."""
        if self._client is None:
            return None
        try:
            raw = await self._client.get(key)
            return json.loads(raw) if raw is not None else None
        except Exception as exc:
            logger.debug("Redis GET error [%s]: %s", key, exc)
            return None

    async def set(
        self,
        key:   str,
        value: Any,
        ttl:   int = 60,
    ) -> bool:
        """Set a JSON value with optional TTL (seconds)."""
        if self._client is None:
            return False
        try:
            await self._client.set(key, json.dumps(value), ex=ttl)
            return True
        except Exception as exc:
            logger.debug("Redis SET error [%s]: %s", key, exc)
            return False

    async def delete(self, key: str) -> bool:
        """Delete a key."""
        if self._client is None:
            return False
        try:
            await self._client.delete(key)
            return True
        except Exception:
            return False

    async def exists(self, key: str) -> bool:
        """Return True if the key exists."""
        if self._client is None:
            return False
        try:
            return bool(await self._client.exists(key))
        except Exception:
            return False

    async def incr(self, key: str) -> int:
        """Atomically increment an integer counter."""
        if self._client is None:
            return 0
        try:
            return await self._client.incr(key)
        except Exception:
            return 0

    async def ping(self) -> bool:
        """Return True if Redis is reachable."""
        if self._client is None:
            return False
        try:
            return await self._client.ping()
        except Exception:
            return False

    async def close(self) -> None:
        """Close the connection pool."""
        if self._client is not None:
            await self._client.close()

    def __repr__(self) -> str:
        return f"RedisClient(url={self.url!r}, available={self._client is not None})"


# ── Module-level singleton ────────────────────────────────────────────────────
_instance: Optional[RedisClient] = None


def get_redis() -> RedisClient:
    """Return the module-level Redis client singleton."""
    global _instance
    if _instance is None:
        _instance = RedisClient()
    return _instance