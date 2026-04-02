"""
drift/mmd_detector.py
======================
Maximum Mean Discrepancy (MMD) concept drift detector.

Mathematical background
-----------------------
MMD is a kernel-based measure of distance between two distributions.
For samples X = {x_1,...,x_n} from P and Y = {y_1,...,y_m} from Q:

    MMD²(P, Q) = E_{x,x'~P}[k(x,x')] - 2·E_{x~P,y~Q}[k(x,y)] + E_{y,y'~Q}[k(y,y')]

where k is a positive-definite kernel (we use RBF / Gaussian):

    k(x, y) = exp(-||x - y||² / (2σ²))

The unbiased estimator for MMD² (Gretton et al., 2012):

    MMD²_u = (1/n(n-1)) Σ_{i≠j} k(x_i,x_j)
           - (2/nm)     Σ_{i,j}  k(x_i,y_j)
           + (1/m(m-1)) Σ_{i≠j} k(y_i,y_j)

MMD² = 0 iff P = Q (for a characteristic kernel like RBF).
MMD² > 0 indicates P ≠ Q.

We take MMD = sqrt(max(MMD², 0)) as the final distance metric.

Kernel bandwidth σ
------------------
The bandwidth σ controls sensitivity:
    - σ too small: sensitive to noise, high false-positive rate
    - σ too large: insensitive to distributional shifts
    - σ = median(pairwise distances) (median heuristic) — our default

Computational complexity
------------------------
Naïve: O(n²) — expensive for large windows.
We use subsampling to cap at 500 samples per window for fast computation.

Compared to Wasserstein and KS:
    + MMD captures higher-order distributional differences (beyond marginals)
    + MMD handles multivariate dependencies between dimensions
    - MMD is slower (O(n²) vs O(n log n) for Wasserstein)
    - MMD threshold is less interpretable than KS p-values

Default threshold: 0.02 (validated on CybORG Scenario2 trajectories).

Usage
-----
    detector = MMDDetector(threshold=0.02, bandwidth="median")
    detector.add_observation(obs)
    result = detector.check(step=global_step)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Union

import numpy as np

from drift.base_detector import BaseDetector

logger = logging.getLogger(__name__)

_DEFAULT_MMD_THRESHOLD  = 0.02
_DEFAULT_WINDOW_SIZE    = 500
_DEFAULT_COOLDOWN       = 300
_DEFAULT_MAX_SUBSAMPLE  = 500    # Cap samples for O(n²) efficiency


class MMDDetector(BaseDetector):
    """
    Maximum Mean Discrepancy concept drift detector.

    Uses an RBF kernel with configurable bandwidth.  Kernel bandwidth
    defaults to the median heuristic computed from the reference window.

    Parameters
    ----------
    threshold : float
        MMD distance threshold.  Default 0.02.
    window_size : int
        Observations per window.  Default 500.
    cooldown_steps : int
        Minimum steps between events.  Default 300.
    bandwidth : float or "median"
        RBF kernel bandwidth σ.  ``"median"`` (default) sets σ to the
        median of pairwise distances within the reference window.
    max_subsample : int
        Maximum observations subsampled per window for MMD computation.
        Default 500 (caps O(n²) cost).
    biased : bool
        If False (default), use the unbiased MMD estimator.
        If True, use the biased estimator (faster but biased for small n).
    """

    def __init__(
        self,
        threshold:    float                = _DEFAULT_MMD_THRESHOLD,
        window_size:  int                  = _DEFAULT_WINDOW_SIZE,
        cooldown_steps: int                = _DEFAULT_COOLDOWN,
        bandwidth:    Union[float, str]    = "median",
        max_subsample: int                 = _DEFAULT_MAX_SUBSAMPLE,
        biased:       bool                 = False,
        **kwargs,
    ) -> None:
        super().__init__(
            threshold      = threshold,
            window_size    = window_size,
            cooldown_steps = cooldown_steps,
            **kwargs,
        )
        self.bandwidth     = bandwidth
        self.max_subsample = max_subsample
        self.biased        = biased

        # Cached bandwidth value (recomputed when reference window changes)
        self._cached_bandwidth: Optional[float] = None
        self._rng = np.random.default_rng(42)

    @property
    def detector_type(self) -> str:
        return "MMD"

    # ------------------------------------------------------------------ #
    # Core computation
    # ------------------------------------------------------------------ #

    def compute_distance(
        self,
        ref_window: np.ndarray,
        cur_window: np.ndarray,
    ) -> float:
        """
        Compute MMD distance between reference and current windows.

        Parameters
        ----------
        ref_window : np.ndarray
            Shape (n_ref, obs_dim).
        cur_window : np.ndarray
            Shape (n_cur, obs_dim).

        Returns
        -------
        float
            MMD distance ∈ [0, ∞).
        """
        # Subsample for computational efficiency
        ref = self._subsample(ref_window)
        cur = self._subsample(cur_window)

        # Resolve bandwidth
        sigma = self._resolve_bandwidth(ref)

        # Compute MMD²
        mmd_squared = self._mmd_squared(ref, cur, sigma)

        # Return MMD = sqrt(max(MMD², 0))
        return float(np.sqrt(max(mmd_squared, 0.0)))

    # ------------------------------------------------------------------ #
    # MMD computation methods
    # ------------------------------------------------------------------ #

    def _mmd_squared(
        self,
        X:     np.ndarray,
        Y:     np.ndarray,
        sigma: float,
    ) -> float:
        """
        Compute the (unbiased or biased) MMD² estimate.

        Parameters
        ----------
        X : np.ndarray
            Reference samples, shape (n, d).
        Y : np.ndarray
            Current samples, shape (m, d).
        sigma : float
            RBF kernel bandwidth.

        Returns
        -------
        float
            MMD² estimate.
        """
        K_XX = self._rbf_kernel_matrix(X, X, sigma)
        K_YY = self._rbf_kernel_matrix(Y, Y, sigma)
        K_XY = self._rbf_kernel_matrix(X, Y, sigma)

        n = len(X)
        m = len(Y)

        if self.biased:
            # Biased estimator (includes diagonal)
            mmd2 = (
                np.mean(K_XX)
                - 2.0 * np.mean(K_XY)
                + np.mean(K_YY)
            )
        else:
            # Unbiased estimator (excludes diagonal)
            # Remove diagonal from K_XX and K_YY
            np.fill_diagonal(K_XX, 0.0)
            np.fill_diagonal(K_YY, 0.0)

            mmd2 = (
                np.sum(K_XX) / max(n * (n - 1), 1)
                - 2.0 * np.mean(K_XY)
                + np.sum(K_YY) / max(m * (m - 1), 1)
            )

        return float(mmd2)

    @staticmethod
    def _rbf_kernel_matrix(
        X:     np.ndarray,
        Y:     np.ndarray,
        sigma: float,
    ) -> np.ndarray:
        """
        Compute the RBF (Gaussian) kernel matrix K[i,j] = k(x_i, y_j).

        k(x, y) = exp(-||x - y||² / (2σ²))

        Parameters
        ----------
        X : np.ndarray
            Shape (n, d).
        Y : np.ndarray
            Shape (m, d).
        sigma : float
            Kernel bandwidth.

        Returns
        -------
        np.ndarray
            Shape (n, m).
        """
        # Efficient pairwise squared distances using broadcast
        # ||x - y||² = ||x||² - 2·x·y^T + ||y||²
        XX = np.sum(X ** 2, axis=1, keepdims=True)  # (n, 1)
        YY = np.sum(Y ** 2, axis=1, keepdims=True)  # (m, 1)
        XY = X @ Y.T                                 # (n, m)

        sq_dists = XX - 2 * XY + YY.T               # (n, m)
        sq_dists = np.clip(sq_dists, 0, None)        # numerical safety

        return np.exp(-sq_dists / (2.0 * sigma ** 2 + 1e-12))

    # ------------------------------------------------------------------ #
    # Bandwidth helpers
    # ------------------------------------------------------------------ #

    def _resolve_bandwidth(self, ref: np.ndarray) -> float:
        """
        Resolve the kernel bandwidth σ.

        If ``self.bandwidth == "median"``, computes the median heuristic:
            σ = median(||x_i - x_j||)  for all i ≠ j

        Otherwise, returns the configured float value.

        The result is cached and invalidated when the reference window
        changes (after each drift event).
        """
        if isinstance(self.bandwidth, (int, float)):
            return float(self.bandwidth)

        # Median heuristic
        if self._cached_bandwidth is not None:
            return self._cached_bandwidth

        self._cached_bandwidth = self._median_bandwidth(ref)
        logger.debug(
            "MMDDetector: median bandwidth σ = %.4f", self._cached_bandwidth
        )
        return self._cached_bandwidth

    @staticmethod
    def _median_bandwidth(X: np.ndarray) -> float:
        """
        Compute σ = median pairwise distance of X.

        Uses a random subsample of 200 points for efficiency.
        """
        n = min(len(X), 200)
        idx = np.random.choice(len(X), size=n, replace=False)
        X_sub = X[idx]

        # Pairwise squared distances
        XX = np.sum(X_sub ** 2, axis=1, keepdims=True)
        sq_dists = XX - 2 * (X_sub @ X_sub.T) + XX.T
        sq_dists = np.clip(sq_dists, 0, None)

        # Extract upper triangle (exclude diagonal)
        upper = sq_dists[np.triu_indices(n, k=1)]
        if len(upper) == 0:
            return 1.0

        median_sq_dist = float(np.median(upper))
        sigma = float(np.sqrt(median_sq_dist + 1e-12))
        return max(sigma, 1e-6)

    def _subsample(self, window: np.ndarray) -> np.ndarray:
        """
        Randomly subsample up to ``self.max_subsample`` rows.

        Parameters
        ----------
        window : np.ndarray
            Shape (n, d).

        Returns
        -------
        np.ndarray
            Shape (min(n, max_subsample), d).
        """
        n = len(window)
        if n <= self.max_subsample:
            return window
        idx = self._rng.choice(n, size=self.max_subsample, replace=False)
        return window[idx]

    # ------------------------------------------------------------------ #
    # Post-drift cache invalidation
    # ------------------------------------------------------------------ #

    def promote_current_to_reference(self) -> None:
        """Override to also invalidate the cached bandwidth."""
        super()._window_manager.promote_current_to_reference()
        self._cached_bandwidth = None

    # ------------------------------------------------------------------ #
    # Extended metadata
    # ------------------------------------------------------------------ #

    def _build_event_metadata(
        self,
        ref_win:  np.ndarray,
        cur_win:  np.ndarray,
        distance: float,
    ) -> Dict[str, Any]:
        meta = super()._build_event_metadata(ref_win, cur_win, distance)
        meta["mmd_distance"] = round(distance, 6)
        meta["mmd_squared"]  = round(distance ** 2, 6)
        meta["bandwidth"]    = (
            round(self._cached_bandwidth, 4)
            if self._cached_bandwidth is not None
            else str(self.bandwidth)
        )
        meta["biased_estimator"] = self.biased
        meta["subsampled_to"]    = self.max_subsample
        return meta

    def __repr__(self) -> str:
        bw = (
            f"σ={self._cached_bandwidth:.4f}"
            if self._cached_bandwidth is not None
            else f"bandwidth={self.bandwidth!r}"
        )
        return (
            f"MMDDetector("
            f"threshold={self.threshold}, "
            f"{bw}, "
            f"events={self.n_events})"
        )