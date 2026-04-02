"""
drift/base_detector.py
=======================
Abstract base class for all concept drift detectors in the ACD Framework.

Every concrete detector (Wasserstein, KS, MMD) inherits from
``BaseDetector`` and implements two abstract methods:

    compute_distance(ref_window, cur_window) → float
        Computes the distributional distance between two observation windows.

    is_drift(distance) → bool
        Applies the threshold rule to declare drift.

The base class handles:
    - Observation feeding and window management delegation
    - Cooldown period enforcement (debouncing rapid drift events)
    - Drift event logging and history
    - Standardised metrics for the API and TensorBoard

Usage
-----
    class MyDetector(BaseDetector):
        def compute_distance(self, ref, cur): ...
        def is_drift(self, distance): ...

    detector = MyDetector(threshold=0.15, window_size=1000)
    detector.add_observation(obs_vector)
    result = detector.check()
    if result.drift_detected:
        print(f"Drift! Score={result.distance:.4f}")
"""

from __future__ import annotations

import abc
import logging
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Default detector parameters
_DEFAULT_THRESHOLD   = 0.15
_DEFAULT_WINDOW_SIZE = 1000
_DEFAULT_COOLDOWN    = 500    # minimum steps between drift events


@dataclass
class DriftEvent:
    """
    Immutable record of a single detected drift event.

    Attributes
    ----------
    event_id : int
        Sequential event counter (1-indexed).
    step : int
        Global training step at which drift was detected.
    distance : float
        Computed distributional distance that triggered drift.
    threshold : float
        Threshold that was exceeded.
    detector_type : str
        Name of the detector that raised this event.
    timestamp : float
        Unix timestamp of detection.
    metadata : dict
        Additional detector-specific details.
    """
    event_id:      int
    step:          int
    distance:      float
    threshold:     float
    detector_type: str
    timestamp:     float                  = field(default_factory=time.time)
    metadata:      Dict[str, Any]         = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id":      self.event_id,
            "step":          self.step,
            "distance":      round(self.distance, 6),
            "threshold":     round(self.threshold, 6),
            "detector_type": self.detector_type,
            "timestamp":     self.timestamp,
            "metadata":      self.metadata,
        }

    def __str__(self) -> str:
        return (
            f"DriftEvent(#{self.event_id}, step={self.step}, "
            f"distance={self.distance:.4f} > threshold={self.threshold:.4f}, "
            f"detector={self.detector_type})"
        )


@dataclass
class DriftResult:
    """
    Result of a single drift check call.

    Attributes
    ----------
    drift_detected : bool
        True if drift was declared (distance > threshold AND cooldown passed).
    distance : float
        Computed distance for this check.
    threshold : float
        Threshold used.
    step : int
        Global step at which the check was performed.
    in_cooldown : bool
        True if drift was detected but suppressed by cooldown.
    window_ready : bool
        True if both windows have sufficient observations.
    event : DriftEvent, optional
        The DriftEvent created if drift_detected is True.
    """
    drift_detected: bool
    distance:       float
    threshold:      float
    step:           int
    in_cooldown:    bool                  = False
    window_ready:   bool                  = True
    event:          Optional[DriftEvent]  = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "drift_detected": self.drift_detected,
            "distance":       round(self.distance, 6),
            "threshold":      round(self.threshold, 6),
            "step":           self.step,
            "in_cooldown":    self.in_cooldown,
            "window_ready":   self.window_ready,
            "event":          self.event.to_dict() if self.event else None,
        }

    def __bool__(self) -> bool:
        return self.drift_detected


class BaseDetector(abc.ABC):
    """
    Abstract base class for all ACD concept drift detectors.

    Parameters
    ----------
    threshold : float
        Distance threshold above which drift is declared.
        Detector-specific: Wasserstein uses ~0.15, KS uses ~0.05,
        MMD uses ~0.02.
    window_size : int
        Number of observations per window.  Both reference and current
        windows use this size.
    cooldown_steps : int
        Minimum steps between consecutive drift events.
        Prevents rapid-fire drift declarations on a sustained shift.
    min_observations : int
        Minimum observations required in each window before drift can
        be declared.  Default: window_size // 2.
    on_drift : callable, optional
        Callback fired when drift is detected.
        Signature: ``(DriftEvent) -> None``.
    """

    def __init__(
        self,
        threshold:        float    = _DEFAULT_THRESHOLD,
        window_size:      int      = _DEFAULT_WINDOW_SIZE,
        cooldown_steps:   int      = _DEFAULT_COOLDOWN,
        min_observations: Optional[int] = None,
        on_drift:         Optional[Callable[[DriftEvent], None]] = None,
    ) -> None:
        self.threshold        = threshold
        self.window_size      = window_size
        self.cooldown_steps   = cooldown_steps
        self.min_observations = min_observations or window_size // 2
        self._on_drift        = on_drift

        # Window manager (initialised per subclass or shared)
        from drift.window_manager import WindowManager
        self._window_manager  = WindowManager(window_size=window_size)

        # State
        self._global_step:      int             = 0
        self._last_drift_step:  int             = -cooldown_steps   # so first drift always allowed
        self._event_counter:    int             = 0
        self._events:           List[DriftEvent] = []
        self._distance_history: List[float]     = []

    # ------------------------------------------------------------------ #
    # Abstract interface
    # ------------------------------------------------------------------ #

    @abc.abstractmethod
    def compute_distance(
        self,
        ref_window: np.ndarray,
        cur_window: np.ndarray,
    ) -> float:
        """
        Compute the distributional distance between two observation windows.

        Parameters
        ----------
        ref_window : np.ndarray
            Reference (baseline) observations, shape (n_ref, obs_dim).
        cur_window : np.ndarray
            Current observations, shape (n_cur, obs_dim).

        Returns
        -------
        float
            Non-negative distance score.  Higher = more distributional shift.
        """

    @property
    @abc.abstractmethod
    def detector_type(self) -> str:
        """Short string name for this detector (e.g. 'Wasserstein')."""

    # ------------------------------------------------------------------ #
    # Concrete interface
    # ------------------------------------------------------------------ #

    def add_observation(self, obs: np.ndarray) -> None:
        """
        Add a single observation vector to the current window.

        Parameters
        ----------
        obs : np.ndarray
            Observation vector, shape (obs_dim,).
        """
        self._global_step += 1
        self._window_manager.add(obs)

    def add_batch(self, obs_batch: np.ndarray) -> None:
        """
        Add a batch of observation vectors.

        Parameters
        ----------
        obs_batch : np.ndarray
            Shape (batch_size, obs_dim).
        """
        for obs in obs_batch:
            self.add_observation(obs)

    def check(self, step: Optional[int] = None) -> DriftResult:
        """
        Check for drift using the current window contents.

        This is the primary method called from the training loop
        (typically every N steps or at episode boundaries).

        Parameters
        ----------
        step : int, optional
            Global step override.  Uses internal counter if None.

        Returns
        -------
        DriftResult
        """
        effective_step = step if step is not None else self._global_step

        # Check if windows are ready
        ref_win, cur_win = self._window_manager.get_windows()
        if ref_win is None or len(ref_win) < self.min_observations:
            return DriftResult(
                drift_detected = False,
                distance       = 0.0,
                threshold      = self.threshold,
                step           = effective_step,
                window_ready   = False,
            )

        if cur_win is None or len(cur_win) < self.min_observations:
            return DriftResult(
                drift_detected = False,
                distance       = 0.0,
                threshold      = self.threshold,
                step           = effective_step,
                window_ready   = False,
            )

        # Compute distance
        try:
            distance = self.compute_distance(ref_win, cur_win)
        except Exception as exc:
            logger.warning(
                "%s compute_distance failed: %s — returning 0.0",
                self.detector_type, exc,
            )
            distance = 0.0

        self._distance_history.append(distance)
        if len(self._distance_history) > 10_000:
            self._distance_history.pop(0)

        # Check threshold
        raw_drift = distance > self.threshold

        # Check cooldown
        steps_since_last = effective_step - self._last_drift_step
        in_cooldown = steps_since_last < self.cooldown_steps

        if raw_drift and in_cooldown:
            logger.debug(
                "%s drift suppressed by cooldown (steps_since=%d < %d)",
                self.detector_type, steps_since_last, self.cooldown_steps,
            )
            return DriftResult(
                drift_detected = False,
                distance       = distance,
                threshold      = self.threshold,
                step           = effective_step,
                in_cooldown    = True,
                window_ready   = True,
            )

        if raw_drift:
            # Drift confirmed — create event
            self._event_counter += 1
            event = DriftEvent(
                event_id      = self._event_counter,
                step          = effective_step,
                distance      = distance,
                threshold     = self.threshold,
                detector_type = self.detector_type,
                metadata      = self._build_event_metadata(ref_win, cur_win, distance),
            )
            self._events.append(event)
            self._last_drift_step = effective_step

            # Reset reference window to current (slide the baseline forward)
            self._window_manager.promote_current_to_reference()

            # Fire callback
            if self._on_drift is not None:
                try:
                    self._on_drift(event)
                except Exception as exc:
                    logger.warning("on_drift callback raised: %s", exc)

            logger.info(
                "%s drift detected — event #%d, step=%d, "
                "distance=%.4f > threshold=%.4f",
                self.detector_type, self._event_counter,
                effective_step, distance, self.threshold,
            )

            return DriftResult(
                drift_detected = True,
                distance       = distance,
                threshold      = self.threshold,
                step           = effective_step,
                window_ready   = True,
                event          = event,
            )

        return DriftResult(
            drift_detected = False,
            distance       = distance,
            threshold      = self.threshold,
            step           = effective_step,
            window_ready   = True,
        )

    # ------------------------------------------------------------------ #
    # Accessors
    # ------------------------------------------------------------------ #

    @property
    def has_drifted(self) -> bool:
        """True if at least one drift event has been detected."""
        return len(self._events) > 0

    @property
    def n_events(self) -> int:
        """Total number of drift events detected."""
        return len(self._events)

    @property
    def drift_history(self) -> List[Dict[str, Any]]:
        """Full list of drift event dicts for the API."""
        return [e.to_dict() for e in self._events]

    @property
    def current_distance(self) -> float:
        """Most recent distance score (0.0 if no checks performed yet)."""
        return self._distance_history[-1] if self._distance_history else 0.0

    @property
    def distance_history(self) -> List[float]:
        """Full distance history for the drift score chart."""
        return list(self._distance_history)

    def get_metrics(self) -> Dict[str, Any]:
        """
        Return current detector metrics for the API / TensorBoard.

        Returns
        -------
        dict
        """
        history = self._distance_history
        return {
            "detector_type":     self.detector_type,
            "threshold":         self.threshold,
            "window_size":       self.window_size,
            "cooldown_steps":    self.cooldown_steps,
            "global_step":       self._global_step,
            "n_drift_events":    self.n_events,
            "current_distance":  round(self.current_distance, 6),
            "max_distance":      round(max(history), 6) if history else 0.0,
            "mean_distance":     round(float(np.mean(history)), 6) if history else 0.0,
            "last_drift_step":   self._last_drift_step if self.has_drifted else None,
        }

    def reset(self) -> None:
        """
        Full reset — clears windows, history, and event log.

        Call between separate training experiments (not between episodes).
        """
        self._window_manager.reset()
        self._global_step     = 0
        self._last_drift_step = -self.cooldown_steps
        self._event_counter   = 0
        self._events.clear()
        self._distance_history.clear()
        logger.debug("%s detector reset.", self.detector_type)

    # ------------------------------------------------------------------ #
    # Protected helpers
    # ------------------------------------------------------------------ #

    def _build_event_metadata(
        self,
        ref_win:  np.ndarray,
        cur_win:  np.ndarray,
        distance: float,
    ) -> Dict[str, Any]:
        """
        Build metadata dict for the DriftEvent.

        Subclasses can override to add detector-specific metadata.
        """
        return {
            "ref_size":      len(ref_win),
            "cur_size":      len(cur_win),
            "obs_dim":       ref_win.shape[1] if ref_win.ndim > 1 else 1,
            "distance":      round(distance, 6),
            "threshold":     self.threshold,
            "steps_since_last_drift": (
                self._global_step - self._last_drift_step
                if self.has_drifted else self._global_step
            ),
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"threshold={self.threshold}, "
            f"window_size={self.window_size}, "
            f"events={self.n_events})"
        )