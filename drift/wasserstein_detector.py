"""
drift/wasserstein_detector.py
==============================
Wasserstein-1 (Earth Mover's Distance) concept drift detector.

Mathematical background
-----------------------
The Wasserstein-1 distance between two 1D distributions P and Q is:

    W1(P, Q) = ∫ |F_P(x) - F_Q(x)| dx

where F_P and F_Q are the cumulative distribution functions.

For empirical distributions (finite samples), this is computed exactly as:

    W1(P_n, Q_m) = (1/n) Σ |x_{(i)} - y_{(i)}|

where x_{(i)}, y_{(i)} are the sorted samples from P and Q respectively
(after interpolating to equal length if n ≠ m).

For multivariate observations (obs_dim > 1), we compute the mean
Wasserstein distance averaged over all dimensions (marginal-wise):

    W1_multi(P, Q) = (1/d) Σ_{j=1}^{d} W1(P_j, Q_j)

where P_j is the j-th marginal distribution.

This is the primary drift detector used in the ACD paper.
Default threshold = 0.15 (validated on CybORG Scenario2 trajectories).

Usage
-----
    detector = WassersteinDetector(
        threshold   = 0.15,
        window_size = 1000,
    )

    detector.add_observation(obs_vector)
    result = detector.check(step=global_step)

    if result.drift_detected:
        ewc.register_task(experience_buffer)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np

from drift.base_detector import BaseDetector, DriftEvent

logger = logging.getLogger(__name__)

# Default threshold validated on CybORG Scenario2
_DEFAULT_W1_THRESHOLD = 0.15
_DEFAULT_WINDOW_SIZE  = 1000
_DEFAULT_COOLDOWN     = 500


class WassersteinDetector(BaseDetector):
    """
    Concept drift detector based on Wasserstein-1 distance.

    Computes the mean marginal Wasserstein-1 distance between a reference
    window and the current sliding window.

    Parameters
    ----------
    threshold : float
        W1 distance threshold.  Default 0.15.
        Tuning guidance:
            - Lower (0.05–0.10): More sensitive, may false-positive on noise.
            - Higher (0.20–0.30): Less sensitive, only catches major shifts.
    window_size : int
        Observations per window.  Default 1000.
    cooldown_steps : int
        Minimum steps between drift events.  Default 500.
    use_pca : bool
        If True, apply PCA to reduce obs_dim before computing distance.
        Useful for high-dimensional observations (obs_dim > 100).
    n_pca_components : int
        Number of PCA components to retain.  Default 10.
    """

    def __init__(
        self,
        threshold:        float = _DEFAULT_W1_THRESHOLD,
        window_size:      int   = _DEFAULT_WINDOW_SIZE,
        cooldown_steps:   int   = _DEFAULT_COOLDOWN,
        use_pca:          bool  = False,
        n_pca_components: int   = 10,
        **kwargs,
    ) -> None:
        super().__init__(
            threshold      = threshold,
            window_size    = window_size,
            cooldown_steps = cooldown_steps,
            **kwargs,
        )
        self.use_pca          = use_pca
        self.n_pca_components = n_pca_components
        self._pca             = None   # fitted sklearn PCA (if use_pca=True)

    @property
    def detector_type(self) -> str:
        return "Wasserstein"

    # ------------------------------------------------------------------ #
    # Core computation
    # ------------------------------------------------------------------ #

    def compute_distance(
        self,
        ref_window: np.ndarray,
        cur_window: np.ndarray,
    ) -> float:
        """
        Compute mean marginal Wasserstein-1 distance.

        Parameters
        ----------
        ref_window : np.ndarray
            Shape (n_ref, obs_dim).
        cur_window : np.ndarray
            Shape (n_cur, obs_dim).

        Returns
        -------
        float
            Mean W1 distance across all observation dimensions.
        """
        # Optionally apply PCA dimensionality reduction
        if self.use_pca:
            ref_window, cur_window = self._apply_pca(ref_window, cur_window)

        return self._mean_marginal_wasserstein(ref_window, cur_window)

    # ------------------------------------------------------------------ #
    # Wasserstein implementations
    # ------------------------------------------------------------------ #

    @staticmethod
    def _mean_marginal_wasserstein(
        ref: np.ndarray,
        cur: np.ndarray,
    ) -> float:
        """
        Compute the mean Wasserstein-1 distance across all marginals.

        For each dimension j, sort both samples and compute the mean
        absolute difference between their empirical CDFs.

        Parameters
        ----------
        ref : np.ndarray
            Reference observations, shape (n_ref, d).
        cur : np.ndarray
            Current observations, shape (n_cur, d).

        Returns
        -------
        float
            Mean W1 distance ∈ [0, ∞).
        """
        if ref.ndim == 1:
            ref = ref.reshape(-1, 1)
        if cur.ndim == 1:
            cur = cur.reshape(-1, 1)

        n_dim = ref.shape[1]
        distances = np.empty(n_dim, dtype=np.float64)

        for j in range(n_dim):
            distances[j] = WassersteinDetector._w1_1d(ref[:, j], cur[:, j])

        return float(np.mean(distances))

    @staticmethod
    def _w1_1d(ref: np.ndarray, cur: np.ndarray) -> float:
        """
        Compute Wasserstein-1 between two 1D empirical distributions.

        Uses the sorted CDF formula:
            W1 = mean(|F_ref^{-1}(t) - F_cur^{-1}(t)|)

        Parameters
        ----------
        ref : np.ndarray
            1D reference samples.
        cur : np.ndarray
            1D current samples.

        Returns
        -------
        float
        """
        n = max(len(ref), len(cur))

        # Interpolate both to the same number of quantile points
        ref_sorted = np.sort(ref)
        cur_sorted = np.sort(cur)

        # Resample both to n points via linear interpolation
        ref_q = np.interp(
            np.linspace(0, 1, n),
            np.linspace(0, 1, len(ref_sorted)),
            ref_sorted,
        )
        cur_q = np.interp(
            np.linspace(0, 1, n),
            np.linspace(0, 1, len(cur_sorted)),
            cur_sorted,
        )

        return float(np.mean(np.abs(ref_q - cur_q)))

    @staticmethod
    def _w1_scipy(ref: np.ndarray, cur: np.ndarray) -> float:
        """
        Compute Wasserstein-1 using scipy if available (more accurate).

        Falls back to the sorted CDF formula if scipy is unavailable.
        """
        try:
            from scipy.stats import wasserstein_distance
            return float(wasserstein_distance(ref, cur))
        except ImportError:
            return WassersteinDetector._w1_1d(ref, cur)

    # ------------------------------------------------------------------ #
    # PCA helper
    # ------------------------------------------------------------------ #

    def _apply_pca(
        self,
        ref: np.ndarray,
        cur: np.ndarray,
    ):
        """
        Fit PCA on the reference window and project both windows.

        PCA is re-fitted each time the reference window changes
        (i.e. after each drift event).
        """
        try:
            from sklearn.decomposition import PCA

            n_components = min(
                self.n_pca_components,
                ref.shape[1],
                ref.shape[0] - 1,
            )

            if self._pca is None or self._pca.n_components_ != n_components:
                self._pca = PCA(n_components=n_components)
                self._pca.fit(ref)

            ref_proj = self._pca.transform(ref)
            cur_proj = self._pca.transform(cur)
            return ref_proj, cur_proj

        except ImportError:
            logger.warning(
                "sklearn not installed — PCA disabled. "
                "Install with: pip install scikit-learn"
            )
            self.use_pca = False
            return ref, cur

    # ------------------------------------------------------------------ #
    # Extended event metadata
    # ------------------------------------------------------------------ #

    def _build_event_metadata(
        self,
        ref_win:  np.ndarray,
        cur_win:  np.ndarray,
        distance: float,
    ) -> Dict[str, Any]:
        meta = super()._build_event_metadata(ref_win, cur_win, distance)

        # Add per-dimension breakdown for the most shifted dimensions
        per_dim = [
            self._w1_1d(ref_win[:, j], cur_win[:, j])
            for j in range(min(ref_win.shape[1], 54))
        ]
        top_dims = sorted(
            enumerate(per_dim), key=lambda x: -x[1]
        )[:5]

        meta["wasserstein_distance"]    = round(distance, 6)
        meta["top_shifted_dims"]        = [
            {"dim": d, "w1": round(w, 4)} for d, w in top_dims
        ]
        meta["use_pca"]                 = self.use_pca
        return meta

    def get_per_dimension_distances(self) -> np.ndarray:
        """
        Compute per-dimension Wasserstein distances for the current windows.

        Returns
        -------
        np.ndarray
            Shape (obs_dim,).  W1 distance per observation dimension.
            Returns zeros if windows not ready.
        """
        ref, cur = self._window_manager.get_windows()
        if ref is None or cur is None:
            return np.zeros(54, dtype=np.float32)

        n_dim = min(ref.shape[1], cur.shape[1])
        dists = np.array([
            self._w1_1d(ref[:, j], cur[:, j]) for j in range(n_dim)
        ], dtype=np.float32)
        return dists