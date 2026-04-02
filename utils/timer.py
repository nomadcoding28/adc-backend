"""
utils/timer.py
===============
Context manager and decorator for measuring elapsed time.

Usage
-----
    from utils.timer import Timer

    # Context manager
    with Timer("KG rebuild") as t:
        builder.build_full()
    print(f"Elapsed: {t.elapsed_s:.2f}s")

    # Decorator
    @Timer.measure
    def my_function():
        ...
"""

from __future__ import annotations

import functools
import logging
import time
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class Timer:
    """
    Context manager that measures wall-clock elapsed time.

    Attributes
    ----------
    elapsed_s : float
        Elapsed time in seconds (set on context exit).
    elapsed_ms : float
        Elapsed time in milliseconds.
    """

    def __init__(self, name: str = "", log: bool = True) -> None:
        self.name      = name
        self.log       = log
        self.elapsed_s = 0.0
        self._start:   Optional[float] = None

    def __enter__(self) -> "Timer":
        self._start = time.monotonic()
        return self

    def __exit__(self, *_: Any) -> None:
        self.elapsed_s = time.monotonic() - (self._start or 0)
        if self.log and self.name:
            logger.debug(
                "%s completed in %.3fs", self.name, self.elapsed_s
            )

    @property
    def elapsed_ms(self) -> float:
        return self.elapsed_s * 1000

    def __repr__(self) -> str:
        return f"Timer({self.name!r}, elapsed={self.elapsed_s:.3f}s)"

    @staticmethod
    def measure(func: Callable) -> Callable:
        """Decorator that logs the execution time of a function."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            with Timer(name=func.__qualname__, log=True):
                return func(*args, **kwargs)
        return wrapper

    @staticmethod
    def measure_async(func: Callable) -> Callable:
        """Decorator for async functions."""
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            start  = time.monotonic()
            result = await func(*args, **kwargs)
            elapsed= time.monotonic() - start
            logger.debug("%s completed in %.3fs", func.__qualname__, elapsed)
            return result
        return wrapper