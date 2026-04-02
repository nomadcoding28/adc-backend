"""
utils/metrics_tracker.py
=========================
Rolling window statistics tracker for training metrics.

Tracks mean, std, min, max over a configurable sliding window.
Used by the CVaR-PPO agent to compute running reward statistics.

Usage
-----
    tracker = RollingMetrics(window=100)

    for episode_reward in rewards:
        tracker.update("episode_reward", episode_reward)

    print(tracker.mean("episode_reward"))   # mean over last 100 values
    print(tracker.get_all())               # all tracked metrics as dict
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any, Deque, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class RollingMetrics:
    """
    Tracks rolling statistics for named scalar metrics.

    Parameters
    ----------
    window : int
        Size of the rolling window.  Default 100.
    """

    def __init__(self, window: int = 100) -> None:
        self.window = window
        self._buffers: Dict[str, Deque[float]] = {}

    def update(self, name: str, value: float) -> None:
        """
        Add a new value to a named metric's rolling window.

        Parameters
        ----------
        name : str
            Metric name.
        value : float
            New scalar value.
        """
        if name not in self._buffers:
            self._buffers[name] = deque(maxlen=self.window)
        self._buffers[name].append(float(value))

    def update_many(self, values: Dict[str, float]) -> None:
        """Update multiple metrics at once."""
        for name, val in values.items():
            self.update(name, val)

    # ── Statistics ──────────────────────────────────────────────────────

    def mean(self, name: str) -> Optional[float]:
        """Rolling mean of the last ``window`` values."""
        buf = self._buffers.get(name)
        return float(np.mean(buf)) if buf else None

    def std(self, name: str) -> Optional[float]:
        """Rolling standard deviation."""
        buf = self._buffers.get(name)
        return float(np.std(buf)) if buf else None

    def min(self, name: str) -> Optional[float]:
        """Rolling minimum."""
        buf = self._buffers.get(name)
        return float(np.min(buf)) if buf else None

    def max(self, name: str) -> Optional[float]:
        """Rolling maximum."""
        buf = self._buffers.get(name)
        return float(np.max(buf)) if buf else None

    def latest(self, name: str) -> Optional[float]:
        """Most recently added value."""
        buf = self._buffers.get(name)
        return buf[-1] if buf else None

    def count(self, name: str) -> int:
        """Number of values currently in the window."""
        buf = self._buffers.get(name)
        return len(buf) if buf else 0

    def cvar(self, name: str, alpha: float = 0.05) -> Optional[float]:
        """
        Conditional Value-at-Risk of the rolling window.

        E[X | X ≤ VaR_α(X)]

        Parameters
        ----------
        alpha : float
            Risk level.  Default 0.05.
        """
        buf = self._buffers.get(name)
        if not buf:
            return None
        arr    = np.array(buf)
        n_tail = max(1, int(alpha * len(arr)))
        sorted_arr = np.sort(arr)
        return float(np.mean(sorted_arr[:n_tail]))

    # ── Bulk accessors ──────────────────────────────────────────────────

    def get(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Return all statistics for a single metric.

        Returns
        -------
        dict or None
            ``{mean, std, min, max, latest, count}``
        """
        if name not in self._buffers:
            return None
        return {
            "mean":   self.mean(name),
            "std":    self.std(name),
            "min":    self.min(name),
            "max":    self.max(name),
            "latest": self.latest(name),
            "count":  self.count(name),
        }

    def get_all(self) -> Dict[str, Dict[str, Any]]:
        """Return statistics for all tracked metrics."""
        return {name: self.get(name) for name in self._buffers}

    def get_means(self) -> Dict[str, Optional[float]]:
        """Return just the rolling means for all metrics."""
        return {name: self.mean(name) for name in self._buffers}

    def metric_names(self) -> List[str]:
        """Return all tracked metric names."""
        return list(self._buffers.keys())

    def reset(self, name: Optional[str] = None) -> None:
        """
        Clear the rolling buffer for a metric (or all metrics).

        Parameters
        ----------
        name : str, optional
            Metric to reset.  If None, resets all metrics.
        """
        if name is None:
            self._buffers.clear()
        elif name in self._buffers:
            self._buffers[name].clear()

    def __repr__(self) -> str:
        return (
            f"RollingMetrics("
            f"window={self.window}, "
            f"metrics={self.metric_names()})"
        )