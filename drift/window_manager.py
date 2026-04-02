"""
drift/window_manager.py
========================
Sliding and tumbling observation window management for drift detection.

The drift detectors need two distinct windows of observations to compare:
    - Reference window  : baseline distribution (collected before last drift)
    - Current window    : recent observations (most recent W steps)

``WindowManager`` maintains both windows with configurable strategies:

    SLIDING   : Current window always contains the last W observations.
                Reference is fixed until drift is declared.

    TUMBLING  : Both windows are non-overlapping fixed-size batches.
                Reference = batch t-1, Current = batch t.

    ADAPTIVE  : Window size grows until drift is declared, then resets.
                Useful when drift frequency is unknown.

Usage
-----
    wm = WindowManager(window_size=1000, strategy="sliding")

    for obs in training_loop:
        wm.add(obs)

    ref_win, cur_win = wm.get_windows()
    if ref_win is not None and cur_win is not None:
        distance = detector.compute_distance(ref_win, cur_win)

    # After drift is confirmed, advance baseline
    wm.promote_current_to_reference()
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Deque, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Window strategy names
STRATEGY_SLIDING  = "sliding"
STRATEGY_TUMBLING = "tumbling"
STRATEGY_ADAPTIVE = "adaptive"

_SUPPORTED_STRATEGIES = {STRATEGY_SLIDING, STRATEGY_TUMBLING, STRATEGY_ADAPTIVE}


class ObservationWindow:
    """
    Fixed-capacity circular buffer for observation vectors.

    Parameters
    ----------
    capacity : int
        Maximum number of observations stored.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity   = capacity
        self._buffer:   Deque[np.ndarray] = deque(maxlen=capacity)

    def add(self, obs: np.ndarray) -> None:
        """Add a single observation vector."""
        self._buffer.append(obs.astype(np.float32))

    def get(self) -> Optional[np.ndarray]:
        """
        Return buffered observations as a 2D array.

        Returns
        -------
        np.ndarray or None
            Shape (n_obs, obs_dim), or None if buffer is empty.
        """
        if not self._buffer:
            return None
        return np.array(self._buffer, dtype=np.float32)

    def clear(self) -> None:
        """Empty the buffer."""
        self._buffer.clear()

    def snapshot(self) -> Optional[np.ndarray]:
        """Return a copy of the current buffer contents."""
        data = self.get()
        return data.copy() if data is not None else None

    @property
    def n_obs(self) -> int:
        """Current number of observations stored."""
        return len(self._buffer)

    @property
    def is_full(self) -> bool:
        """True if the buffer has reached capacity."""
        return len(self._buffer) == self.capacity

    def __len__(self) -> int:
        return len(self._buffer)

    def __repr__(self) -> str:
        return (
            f"ObservationWindow("
            f"n={self.n_obs}/{self.capacity})"
        )


class WindowManager:
    """
    Manages reference and current observation windows for drift detection.

    Parameters
    ----------
    window_size : int
        Number of observations per window.  Default 1000.
    strategy : str
        Window management strategy.  One of ``"sliding"``,
        ``"tumbling"``, ``"adaptive"``.  Default ``"sliding"``.
    overlap_fraction : float
        For ``"sliding"`` strategy: fraction of old observations retained
        when the current window promotes to reference.  Default 0.0
        (no overlap — clean cut).
    """

    def __init__(
        self,
        window_size:       int   = 1000,
        strategy:          str   = STRATEGY_SLIDING,
        overlap_fraction:  float = 0.0,
    ) -> None:
        if strategy not in _SUPPORTED_STRATEGIES:
            raise ValueError(
                f"Unknown strategy {strategy!r}. "
                f"Supported: {_SUPPORTED_STRATEGIES}"
            )

        self.window_size      = window_size
        self.strategy         = strategy
        self.overlap_fraction = overlap_fraction

        # Reference window — baseline distribution
        self._ref_window: ObservationWindow = ObservationWindow(window_size)
        # Current window — recent observations
        self._cur_window: ObservationWindow = ObservationWindow(window_size)

        # Tumbling window: batch index counter
        self._tumbling_batch: int = 0

        # Adaptive window: grows until drift is declared
        self._adaptive_buffer: list = []

        # Counts
        self._total_obs_added: int = 0

    # ------------------------------------------------------------------ #
    # Primary interface
    # ------------------------------------------------------------------ #

    def add(self, obs: np.ndarray) -> None:
        """
        Add a single observation to the appropriate window.

        For ``"sliding"`` strategy:
            - First ``window_size`` observations fill the reference window.
            - Subsequent observations fill the current window (sliding).

        For ``"tumbling"`` strategy:
            - Observations fill batch 0 (reference) then batch 1 (current),
              then promote and reset.

        For ``"adaptive"`` strategy:
            - All observations accumulate until ``promote_current_to_reference``
              is called.
        """
        self._total_obs_added += 1

        if self.strategy == STRATEGY_SLIDING:
            self._add_sliding(obs)
        elif self.strategy == STRATEGY_TUMBLING:
            self._add_tumbling(obs)
        else:
            self._add_adaptive(obs)

    def get_windows(
        self,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Return the reference and current windows as numpy arrays.

        Returns
        -------
        tuple(ref_window, cur_window)
            Each is either an ``np.ndarray`` of shape (n, obs_dim)
            or ``None`` if not yet populated.
        """
        if self.strategy == STRATEGY_ADAPTIVE:
            return self._get_adaptive_windows()

        ref = self._ref_window.get()
        cur = self._cur_window.get()
        return ref, cur

    def promote_current_to_reference(self) -> None:
        """
        After drift is declared, advance the baseline.

        The current window becomes the new reference, and the current
        window is cleared to start collecting the next batch.

        For ``"sliding"`` strategy with overlap:
            The reference retains the last ``overlap_fraction * window_size``
            observations from the promoted current window.
        """
        cur_data = self._cur_window.get()

        if cur_data is None:
            logger.warning("Cannot promote — current window is empty.")
            return

        self._ref_window.clear()

        if self.overlap_fraction > 0:
            # Retain overlap
            n_overlap = int(self.window_size * self.overlap_fraction)
            overlap   = cur_data[-n_overlap:] if n_overlap < len(cur_data) else cur_data
            for obs in overlap:
                self._ref_window.add(obs)
        else:
            # Full promote — copy current into reference
            for obs in cur_data:
                self._ref_window.add(obs)

        self._cur_window.clear()
        self._tumbling_batch = 0

        logger.debug(
            "WindowManager: current promoted to reference "
            "(ref_size=%d, overlap=%.0f%%)",
            self._ref_window.n_obs,
            self.overlap_fraction * 100,
        )

    def reset(self) -> None:
        """Full reset — clear both windows."""
        self._ref_window.clear()
        self._cur_window.clear()
        self._adaptive_buffer.clear()
        self._tumbling_batch = 0
        self._total_obs_added = 0

    # ------------------------------------------------------------------ #
    # Strategy-specific add methods
    # ------------------------------------------------------------------ #

    def _add_sliding(self, obs: np.ndarray) -> None:
        """
        Sliding window add.

        Fill reference window first.  Once reference is full, add to
        the rolling current window (oldest auto-evicted by deque maxlen).
        """
        if not self._ref_window.is_full:
            self._ref_window.add(obs)
        else:
            self._cur_window.add(obs)

    def _add_tumbling(self, obs: np.ndarray) -> None:
        """
        Tumbling (non-overlapping batch) window add.

        Batch 0 → reference.  Batch 1 → current.  When current is full,
        auto-promote so the process repeats.
        """
        if not self._ref_window.is_full:
            self._ref_window.add(obs)
        elif not self._cur_window.is_full:
            self._cur_window.add(obs)
        else:
            # Both full — auto-promote and start new batch
            self.promote_current_to_reference()
            self._cur_window.add(obs)
            self._tumbling_batch += 1

    def _add_adaptive(self, obs: np.ndarray) -> None:
        """
        Adaptive window add — accumulates in a flat list.

        The caller controls when windows are split via
        ``promote_current_to_reference()``.
        """
        self._adaptive_buffer.append(obs.astype(np.float32))

    def _get_adaptive_windows(
        self,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        For adaptive strategy: split the buffer at midpoint.

        Returns the first half as reference, second half as current.
        """
        buf = self._adaptive_buffer
        if len(buf) < 4:
            return None, None

        mid = len(buf) // 2
        ref = np.array(buf[:mid],  dtype=np.float32)
        cur = np.array(buf[mid:],  dtype=np.float32)
        return ref, cur

    # ------------------------------------------------------------------ #
    # Diagnostics
    # ------------------------------------------------------------------ #

    @property
    def ref_size(self) -> int:
        """Current number of observations in the reference window."""
        if self.strategy == STRATEGY_ADAPTIVE:
            return len(self._adaptive_buffer) // 2
        return self._ref_window.n_obs

    @property
    def cur_size(self) -> int:
        """Current number of observations in the current window."""
        if self.strategy == STRATEGY_ADAPTIVE:
            return len(self._adaptive_buffer) - len(self._adaptive_buffer) // 2
        return self._cur_window.n_obs

    @property
    def total_obs_added(self) -> int:
        """Total observations added since last reset."""
        return self._total_obs_added

    @property
    def is_ready(self) -> bool:
        """
        True if both windows have at least ``window_size // 2`` observations.
        """
        min_size = self.window_size // 2
        return self.ref_size >= min_size and self.cur_size >= min_size

    def get_stats(self) -> dict:
        """Return window fill statistics."""
        return {
            "strategy":        self.strategy,
            "window_size":     self.window_size,
            "ref_size":        self.ref_size,
            "cur_size":        self.cur_size,
            "is_ready":        self.is_ready,
            "total_added":     self._total_obs_added,
        }

    def __repr__(self) -> str:
        return (
            f"WindowManager("
            f"strategy={self.strategy!r}, "
            f"ref={self.ref_size}/{self.window_size}, "
            f"cur={self.cur_size}/{self.window_size})"
        )