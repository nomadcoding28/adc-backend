"""
drift/detector_factory.py
==========================
Factory that builds and configures drift detectors from a config dict.

Also exposes ``DriftDetector`` — a high-level façade that wraps one or
more detectors in an ensemble, providing a single ``step()`` interface
called from the training loop.

Ensemble mode
-------------
When ``ensemble=True`` (default), the factory builds all three detectors
(Wasserstein, KS, MMD) and uses a voting rule:

    drift = majority_vote([wasserstein.check(), ks.check(), mmd.check()])

This reduces false positives — a drift event is only raised when at least
``vote_threshold`` detectors agree.

Single detector mode
--------------------
When ``ensemble=False``, only the configured ``method`` is built.
Use ``"wasserstein"`` (recommended) as the primary detector.

Usage
-----
    # From config dict
    detector = DetectorFactory.build(config={
        "method":         "wasserstein",
        "ensemble":       False,
        "threshold":      0.15,
        "window_size":    1000,
        "cooldown_steps": 500,
    })

    # Or build ensemble
    detector = DetectorFactory.build_ensemble(config={
        "vote_threshold": 2,      # 2 of 3 must agree
        "wasserstein": {"threshold": 0.15},
        "ks":          {"alpha": 0.01},
        "mmd":         {"threshold": 0.02},
    })

    # Use in training loop
    for step, obs in training_loop:
        result = detector.step(obs, global_step=step)
        if result.drift_detected:
            continual_learner.handle_drift(result.event)
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from drift.base_detector import BaseDetector, DriftEvent, DriftResult
from drift.wasserstein_detector import WassersteinDetector
from drift.ks_detector import KSDetector
from drift.mmd_detector import MMDDetector

logger = logging.getLogger(__name__)

# Registry mapping method name → detector class
_DETECTOR_REGISTRY: Dict[str, type] = {
    "wasserstein": WassersteinDetector,
    "ks":          KSDetector,
    "mmd":         MMDDetector,
}


class DriftDetector:
    """
    High-level drift detection façade for the ACD training loop.

    Wraps one or more ``BaseDetector`` instances.  Provides a single
    ``step(obs, global_step)`` method that:
        1. Adds the observation to all detectors
        2. Runs a drift check every ``check_frequency`` steps
        3. Returns a ``DriftResult`` (ensemble vote or single detector)

    Parameters
    ----------
    detectors : list[BaseDetector]
        One or more detector instances.
    vote_threshold : int
        Minimum number of detectors that must agree for ensemble drift.
        Default 1 (any detector triggers drift).
    check_frequency : int
        How often to run the drift check (every N steps).
        Default 100.
    on_drift : callable, optional
        Callback fired on confirmed drift.  Signature: ``(DriftEvent) -> None``.
    """

    def __init__(
        self,
        detectors:       List[BaseDetector],
        vote_threshold:  int = 1,
        check_frequency: int = 100,
        on_drift:        Optional[Callable[[DriftEvent], None]] = None,
    ) -> None:
        if not detectors:
            raise ValueError("At least one detector is required.")

        self.detectors       = detectors
        self.vote_threshold  = vote_threshold
        self.check_frequency = check_frequency
        self._on_drift       = on_drift

        self._step_count:    int             = 0
        self._drift_events:  List[DriftEvent] = []

        logger.info(
            "DriftDetector ready — detectors=%s, vote_threshold=%d, "
            "check_freq=%d",
            [d.detector_type for d in detectors],
            vote_threshold,
            check_frequency,
        )

    # ------------------------------------------------------------------ #
    # Primary interface
    # ------------------------------------------------------------------ #

    def step(
        self,
        obs:         np.ndarray,
        global_step: Optional[int] = None,
    ) -> DriftResult:
        """
        Add an observation and optionally run a drift check.

        Parameters
        ----------
        obs : np.ndarray
            Current observation vector, shape (obs_dim,).
        global_step : int, optional
            Global training step for logging.  Uses internal counter if None.

        Returns
        -------
        DriftResult
            ``drift_detected=True`` only on check steps when drift is confirmed.
        """
        self._step_count += 1
        effective_step = global_step if global_step is not None else self._step_count

        # Add to all detectors
        for d in self.detectors:
            d.add_observation(obs)

        # Run check at configured frequency
        if self._step_count % self.check_frequency != 0:
            return DriftResult(
                drift_detected = False,
                distance       = 0.0,
                threshold      = self.detectors[0].threshold,
                step           = effective_step,
                window_ready   = False,
            )

        return self._check(effective_step)

    def force_check(self, global_step: Optional[int] = None) -> DriftResult:
        """
        Force an immediate drift check regardless of ``check_frequency``.

        Useful at episode boundaries or after significant environment events.

        Parameters
        ----------
        global_step : int, optional

        Returns
        -------
        DriftResult
        """
        effective_step = global_step if global_step is not None else self._step_count
        return self._check(effective_step)

    def add_batch(self, obs_batch: np.ndarray) -> None:
        """
        Add a batch of observations without running a check.

        Parameters
        ----------
        obs_batch : np.ndarray
            Shape (batch_size, obs_dim).
        """
        for obs in obs_batch:
            for d in self.detectors:
                d.add_observation(obs)
            self._step_count += 1

    # ------------------------------------------------------------------ #
    # Ensemble voting
    # ------------------------------------------------------------------ #

    def _check(self, step: int) -> DriftResult:
        """Run all detectors and apply the ensemble vote rule."""
        results: List[DriftResult] = []

        for detector in self.detectors:
            result = detector.check(step=step)
            results.append(result)

        # Count how many detectors flagged drift
        n_drift = sum(1 for r in results if r.drift_detected)

        if n_drift >= self.vote_threshold:
            # Use the result from the primary detector (first in list)
            primary_result = next(
                (r for r in results if r.drift_detected), results[0]
            )

            # Log ensemble agreement
            logger.info(
                "Ensemble drift: %d/%d detectors agreed at step %d",
                n_drift, len(self.detectors), step,
            )

            if primary_result.event is not None:
                self._drift_events.append(primary_result.event)
                if self._on_drift is not None:
                    try:
                        self._on_drift(primary_result.event)
                    except Exception as exc:
                        logger.warning("on_drift callback raised: %s", exc)

            return primary_result

        # No consensus drift — return the result with the highest distance
        best = max(results, key=lambda r: r.distance)
        return DriftResult(
            drift_detected = False,
            distance       = best.distance,
            threshold      = best.threshold,
            step           = step,
            window_ready   = best.window_ready,
        )

    # ------------------------------------------------------------------ #
    # Accessors
    # ------------------------------------------------------------------ #

    @property
    def has_drifted(self) -> bool:
        """True if at least one drift event was confirmed."""
        return len(self._drift_events) > 0

    @property
    def n_events(self) -> int:
        """Total confirmed drift events."""
        return len(self._drift_events)

    @property
    def drift_history(self) -> List[Dict[str, Any]]:
        """All confirmed drift events as dicts for the API."""
        return [e.to_dict() for e in self._drift_events]

    @property
    def current_distance(self) -> float:
        """Most recent distance from the primary (first) detector."""
        return self.detectors[0].current_distance if self.detectors else 0.0

    @property
    def distance_history(self) -> List[float]:
        """Distance history from the primary detector."""
        return self.detectors[0].distance_history if self.detectors else []

    def get_metrics(self) -> Dict[str, Any]:
        """Return metrics from all detectors for the API."""
        return {
            "n_detectors":    len(self.detectors),
            "vote_threshold": self.vote_threshold,
            "check_frequency":self.check_frequency,
            "n_drift_events": self.n_events,
            "step_count":     self._step_count,
            "detectors":      [d.get_metrics() for d in self.detectors],
        }

    def reset(self) -> None:
        """Reset all detectors and event history."""
        for d in self.detectors:
            d.reset()
        self._drift_events.clear()
        self._step_count = 0

    def __repr__(self) -> str:
        return (
            f"DriftDetector("
            f"detectors={[d.detector_type for d in self.detectors]}, "
            f"vote={self.vote_threshold}, "
            f"events={self.n_events})"
        )


class DetectorFactory:
    """
    Factory for building ``DriftDetector`` instances from config dicts.
    """

    @classmethod
    def build(cls, config: Optional[Dict[str, Any]] = None) -> DriftDetector:
        """
        Build a ``DriftDetector`` from a config dict.

        Parameters
        ----------
        config : dict, optional
            Keys:
                method          : "wasserstein" | "ks" | "mmd" | "ensemble"
                ensemble        : bool — build all 3 detectors if True
                vote_threshold  : int — votes needed for ensemble drift
                check_frequency : int — steps between checks
                threshold       : float — for single detector
                window_size     : int
                cooldown_steps  : int
                wasserstein     : dict — Wasserstein-specific config
                ks              : dict — KS-specific config
                mmd             : dict — MMD-specific config

        Returns
        -------
        DriftDetector
        """
        cfg = config or {}

        method    = cfg.get("method", "wasserstein").lower()
        ensemble  = cfg.get("ensemble", method == "ensemble")

        check_frequency = cfg.get("check_frequency", 100)
        vote_threshold  = cfg.get("vote_threshold", 1)

        if ensemble or method == "ensemble":
            return cls.build_ensemble(cfg, check_frequency, vote_threshold)

        # Single detector
        detector = cls._build_single(method, cfg)
        return DriftDetector(
            detectors       = [detector],
            vote_threshold  = 1,
            check_frequency = check_frequency,
        )

    @classmethod
    def build_ensemble(
        cls,
        config:          Optional[Dict[str, Any]] = None,
        check_frequency: int = 100,
        vote_threshold:  int = 2,
    ) -> DriftDetector:
        """
        Build an ensemble ``DriftDetector`` with all three methods.

        Parameters
        ----------
        config : dict, optional
            Sub-configs keyed by detector name:
            ``wasserstein``, ``ks``, ``mmd``.
        check_frequency : int
        vote_threshold : int
            Default 2 (majority of 3 must agree).

        Returns
        -------
        DriftDetector
        """
        cfg = config or {}

        detectors = [
            cls._build_single("wasserstein", cfg.get("wasserstein", cfg)),
            cls._build_single("ks",          cfg.get("ks",          cfg)),
            cls._build_single("mmd",         cfg.get("mmd",         cfg)),
        ]

        logger.info(
            "Built ensemble DriftDetector — vote_threshold=%d", vote_threshold
        )

        return DriftDetector(
            detectors       = detectors,
            vote_threshold  = vote_threshold,
            check_frequency = check_frequency,
        )

    @classmethod
    def _build_single(
        cls,
        method: str,
        config: Dict[str, Any],
    ) -> BaseDetector:
        """
        Build a single detector of the given method type.

        Parameters
        ----------
        method : str
            Detector type name.
        config : dict
            Config dict.

        Returns
        -------
        BaseDetector
        """
        if method not in _DETECTOR_REGISTRY:
            raise ValueError(
                f"Unknown drift detection method {method!r}. "
                f"Available: {list(_DETECTOR_REGISTRY.keys())}"
            )

        detector_cls = _DETECTOR_REGISTRY[method]

        # Extract relevant kwargs from config
        kwargs: Dict[str, Any] = {}

        if "threshold" in config:
            kwargs["threshold"] = config["threshold"]
        if "window_size" in config:
            kwargs["window_size"] = config["window_size"]
        if "cooldown_steps" in config:
            kwargs["cooldown_steps"] = config["cooldown_steps"]

        # Method-specific kwargs
        if method == "ks":
            if "alpha" in config:
                kwargs["alpha"] = config["alpha"]
            if "min_dims_drift" in config:
                kwargs["min_dims_drift"] = config["min_dims_drift"]
            if "use_bonferroni" in config:
                kwargs["use_bonferroni"] = config["use_bonferroni"]

        elif method == "mmd":
            if "bandwidth" in config:
                kwargs["bandwidth"] = config["bandwidth"]
            if "max_subsample" in config:
                kwargs["max_subsample"] = config["max_subsample"]
            if "biased" in config:
                kwargs["biased"] = config["biased"]

        elif method == "wasserstein":
            if "use_pca" in config:
                kwargs["use_pca"] = config["use_pca"]
            if "n_pca_components" in config:
                kwargs["n_pca_components"] = config["n_pca_components"]

        logger.debug(
            "Building %s detector with kwargs=%s", method, kwargs
        )
        return detector_cls(**kwargs)

    @classmethod
    def available_methods(cls) -> List[str]:
        """Return list of available drift detection method names."""
        return list(_DETECTOR_REGISTRY.keys()) + ["ensemble"]

    @classmethod
    def register(cls, name: str, detector_cls: type) -> None:
        """
        Register a custom detector class.

        Parameters
        ----------
        name : str
            Method name string.
        detector_cls : type
            A class that inherits from ``BaseDetector``.
        """
        if not issubclass(detector_cls, BaseDetector):
            raise ValueError(
                f"{detector_cls.__name__} must inherit from BaseDetector."
            )
        _DETECTOR_REGISTRY[name] = detector_cls
        logger.debug("Registered drift detector: %r", name)