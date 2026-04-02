"""
cache/decorators.py
====================
Async function cache decorator.

Usage
-----
    from cache.decorators import cached
    from cache.keys import CacheKeys

    @cached(key=CacheKeys.kg_stats, ttl=CacheKeys.TTL_MEDIUM)
    async def get_kg_stats(client):
        return client.get_stats()   # expensive DB call
"""

from __future__ import annotations

import asyncio
import functools
import logging
from typing import Any, Callable, Optional, Union

logger = logging.getLogger(__name__)


def cached(
    key:      Union[str, Callable],
    ttl:      int = 60,
    skip_none: bool = True,
) -> Callable:
    """
    Decorator that caches the return value of an async function in Redis.

    Parameters
    ----------
    key : str or callable
        Cache key string, or a callable that returns one.
        If callable, it is called with the same args as the decorated function.
    ttl : int
        Time-to-live in seconds.  Default 60.
    skip_none : bool
        If True, do not cache None return values.  Default True.

    Example
    -------
        @cached(key="my:key", ttl=300)
        async def expensive_function(arg):
            return await db.query(...)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            from cache.redis_client import get_redis
            redis = get_redis()

            # Resolve cache key
            if callable(key):
                try:
                    cache_key = key(*args, **kwargs)
                except Exception:
                    cache_key = key()
            else:
                cache_key = key

            # Try cache hit
            cached_value = await redis.get(cache_key)
            if cached_value is not None:
                logger.debug("Cache HIT: %s", cache_key)
                return cached_value

            # Cache miss — call the real function
            result = await func(*args, **kwargs)

            # Store result (unless None and skip_none is True)
            if result is not None or not skip_none:
                await redis.set(cache_key, result, ttl=ttl)
                logger.debug("Cache SET: %s (ttl=%ds)", cache_key, ttl)

            return result

        return wrapper
    return decorator


def invalidate(*keys: str) -> Callable:
    """
    Decorator that deletes specified cache keys after the function runs.

    Useful for write operations that should invalidate stale cache entries.

    Example
    -------
        @invalidate(CacheKeys.kg_stats(), CacheKeys.kg_graph())
        async def rebuild_kg():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            result = await func(*args, **kwargs)

            from cache.redis_client import get_redis
            redis = get_redis()
            for key in keys:
                await redis.delete(key)
                logger.debug("Cache INVALIDATE: %s", key)

            return result
        return wrapper
    return decorator