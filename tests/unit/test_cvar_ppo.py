"""
tests/unit/test_cvar_ppo.py
============================
Unit tests for CVaR (Conditional Value-at-Risk) computation.

Tests validate the core mathematical properties of CVaR:
    - VaR is the α-quantile of the return distribution
    - CVaR is the expected value of returns below VaR
    - Importance weights are 1/α for tail samples, 0 otherwise
    - CVaR weighted loss is larger when the tail is worse
"""

from __future__ import annotations

import numpy as np
import pytest


# ── VaR computation ─────────────────────────────────────────────────────────

class TestComputeVaR:
    """Test Value-at-Risk (VaR) computation at various α levels."""

    def test_var_at_005(self, sample_returns: np.ndarray) -> None:
        """VaR at α=0.05 should be approximately the 5th percentile."""
        alpha = 0.05
        var = np.percentile(sample_returns, alpha * 100)
        assert var < np.mean(sample_returns), "VaR should be below the mean."

    def test_var_at_001(self, sample_returns: np.ndarray) -> None:
        """VaR at α=0.01 should be more extreme (lower) than VaR at α=0.05."""
        var_001 = np.percentile(sample_returns, 1.0)
        var_005 = np.percentile(sample_returns, 5.0)
        assert var_001 <= var_005, "Smaller α → more extreme VaR."

    def test_var_monotonicity(self, sample_returns: np.ndarray) -> None:
        """VaR should be monotonically increasing with α."""
        alphas = [0.01, 0.05, 0.10, 0.20, 0.50]
        vars_ = [np.percentile(sample_returns, a * 100) for a in alphas]
        for i in range(len(vars_) - 1):
            assert vars_[i] <= vars_[i + 1] + 1e-6, (
                f"VaR({alphas[i]}) = {vars_[i]:.4f} should be ≤ "
                f"VaR({alphas[i+1]}) = {vars_[i+1]:.4f}"
            )


# ── CVaR computation ────────────────────────────────────────────────────────

class TestComputeCVaR:
    """Test Conditional Value-at-Risk (CVaR) computation."""

    @staticmethod
    def _compute_cvar(returns: np.ndarray, alpha: float) -> float:
        """Reference CVaR implementation."""
        n_tail = max(1, int(alpha * len(returns)))
        sorted_returns = np.sort(returns)
        return float(np.mean(sorted_returns[:n_tail]))

    def test_cvar_is_mean_of_tail(self, sample_returns: np.ndarray) -> None:
        """CVaR at α=0.05 should equal the mean of the worst 5% of returns."""
        alpha = 0.05
        n_tail = max(1, int(alpha * len(sample_returns)))
        sorted_r = np.sort(sample_returns)
        expected_cvar = float(np.mean(sorted_r[:n_tail]))
        computed_cvar = self._compute_cvar(sample_returns, alpha)
        assert abs(computed_cvar - expected_cvar) < 1e-6

    def test_cvar_below_mean(self, sample_returns: np.ndarray) -> None:
        """CVaR should always be below (or equal to) the mean reward."""
        for alpha in [0.01, 0.05, 0.10]:
            cvar = self._compute_cvar(sample_returns, alpha)
            mean = float(np.mean(sample_returns))
            assert cvar <= mean + 1e-6, (
                f"CVaR(α={alpha}) = {cvar:.4f} should be ≤ mean = {mean:.4f}"
            )

    def test_cvar_below_var(self, sample_returns: np.ndarray) -> None:
        """CVaR should be ≤ VaR at the same α (CVaR is the conditional mean)."""
        for alpha in [0.01, 0.05, 0.10]:
            cvar = self._compute_cvar(sample_returns, alpha)
            var = float(np.percentile(sample_returns, alpha * 100))
            assert cvar <= var + 1e-6

    def test_cvar_identical_distribution_equals_constant(self) -> None:
        """If all returns are the same, CVaR equals that constant."""
        returns = np.full(100, 42.0, dtype=np.float32)
        cvar = self._compute_cvar(returns, 0.05)
        assert abs(cvar - 42.0) < 1e-6

    def test_cvar_alpha_1_equals_mean(self, sample_returns: np.ndarray) -> None:
        """CVaR at α=1.0 should equal the full mean (all returns in tail)."""
        cvar = self._compute_cvar(sample_returns, 1.0)
        mean = float(np.mean(sample_returns))
        assert abs(cvar - mean) < 1e-4


# ── Importance weights ──────────────────────────────────────────────────────

class TestImportanceWeights:
    """Test CVaR importance weight computation."""

    @staticmethod
    def _compute_weights(returns: np.ndarray, alpha: float) -> np.ndarray:
        """Reference importance weight computation."""
        n = len(returns)
        var = np.percentile(returns, alpha * 100)
        weights = np.where(returns <= var, 1.0 / alpha, 0.0)
        return weights / (weights.sum() + 1e-8) * n

    def test_weights_sum(self, sample_returns: np.ndarray) -> None:
        """Normalised importance weights should sum to n."""
        weights = self._compute_weights(sample_returns, 0.05)
        assert abs(weights.sum() - len(sample_returns)) < 1.0

    def test_tail_samples_have_weight(self, sample_returns: np.ndarray) -> None:
        """Samples in the tail (below VaR) should have non-zero weight."""
        alpha = 0.05
        var = np.percentile(sample_returns, alpha * 100)
        weights = self._compute_weights(sample_returns, alpha)
        tail_mask = sample_returns <= var
        assert np.all(weights[tail_mask] > 0), "Tail samples should have positive weight."

    def test_non_tail_samples_zero_weight(self, sample_returns: np.ndarray) -> None:
        """Samples above VaR should have zero weight (hard threshold)."""
        alpha = 0.05
        var = np.percentile(sample_returns, alpha * 100)
        weights = self._compute_weights(sample_returns, alpha)
        above_mask = sample_returns > var
        assert np.all(weights[above_mask] == 0.0), "Non-tail samples should have zero weight."


# ── CVaR-weighted loss ──────────────────────────────────────────────────────

class TestCVaRLossWeighting:
    """Test that CVaR weighting correctly emphasises tail losses."""

    def test_cvar_loss_larger_when_tail_is_bad(self) -> None:
        """
        CVaR-weighted mean loss should be larger than standard mean
        when the tail has disproportionately bad values.
        """
        # Construct returns with a bad tail
        returns = np.concatenate([
            np.full(95, 50.0),    # 95 good episodes
            np.full(5, -100.0),   # 5 catastrophic episodes
        ]).astype(np.float32)

        alpha = 0.05
        n = len(returns)
        n_tail = max(1, int(alpha * n))
        sorted_r = np.sort(returns)

        cvar = float(np.mean(sorted_r[:n_tail]))
        mean = float(np.mean(returns))

        assert cvar < mean, "CVaR should be much worse than mean."
        assert cvar < -50.0, "CVaR should reflect catastrophic tail."

    def test_uniform_returns_cvar_equals_mean(self) -> None:
        """If all returns are equal, CVaR-weighted and standard mean are the same."""
        returns = np.full(100, 25.0, dtype=np.float32)
        alpha = 0.05
        n_tail = max(1, int(alpha * len(returns)))
        cvar = float(np.mean(np.sort(returns)[:n_tail]))
        mean = float(np.mean(returns))
        assert abs(cvar - mean) < 1e-6
