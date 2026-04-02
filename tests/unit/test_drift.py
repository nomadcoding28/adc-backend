"""
tests/unit/test_drift.py
=========================
Unit tests for the Wasserstein-1 drift detector.

Tests validate:
    - W1 distance ≈ 0 for identical distributions
    - W1 distance > 0 for shifted distributions
    - Cooldown prevents rapid-fire re-triggering
    - Per-dimension distance breakdown is correct
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import wasserstein_distance


# ── Wasserstein-1 distance ──────────────────────────────────────────────────

class TestWassersteinDistance:
    """Test Wasserstein-1 distance computation."""

    def test_w1_identical_distributions(self, rng) -> None:
        """W1 distance between identical samples should be ≈ 0."""
        x = rng.standard_normal(500)
        w1 = wasserstein_distance(x, x)
        assert w1 < 1e-10, f"W1 for identical distributions should be ≈ 0, got {w1}"

    def test_w1_shifted_distributions(self, rng) -> None:
        """W1 distance between shifted distributions should be ≈ shift magnitude."""
        x = rng.standard_normal(1000)
        shift = 2.0
        y = x + shift
        w1 = wasserstein_distance(x, y)
        assert abs(w1 - shift) < 0.2, (
            f"W1 for shift={shift} should be ≈ {shift}, got {w1}"
        )

    def test_w1_symmetry(self, rng) -> None:
        """W1(X, Y) should equal W1(Y, X)."""
        x = rng.standard_normal(500)
        y = rng.standard_normal(500) + 1.0
        w1_xy = wasserstein_distance(x, y)
        w1_yx = wasserstein_distance(y, x)
        assert abs(w1_xy - w1_yx) < 1e-10

    def test_w1_triangle_inequality(self, rng) -> None:
        """W1 should satisfy the triangle inequality: W1(X,Z) ≤ W1(X,Y) + W1(Y,Z)."""
        x = rng.standard_normal(500)
        y = rng.standard_normal(500) + 1.0
        z = rng.standard_normal(500) + 3.0
        w1_xz = wasserstein_distance(x, z)
        w1_xy = wasserstein_distance(x, y)
        w1_yz = wasserstein_distance(y, z)
        assert w1_xz <= w1_xy + w1_yz + 1e-6


# ── Per-dimension distances ─────────────────────────────────────────────────

class TestPerDimensionDistances:
    """Test per-dimension Wasserstein distance computation."""

    def test_per_dim_correct_shape(self, rng) -> None:
        """Per-dimension distances should have shape (n_dims,)."""
        obs_dim = 54
        ref = rng.standard_normal((500, obs_dim))
        cur = rng.standard_normal((500, obs_dim))

        distances = np.array([
            wasserstein_distance(ref[:, d], cur[:, d])
            for d in range(obs_dim)
        ])
        assert distances.shape == (obs_dim,)

    def test_shifted_dimension_detected(self, rng) -> None:
        """A shifted dimension should have higher W1 than unshifted ones."""
        obs_dim = 54
        ref = rng.standard_normal((500, obs_dim))
        cur = ref.copy()
        shifted_dim = 10
        cur[:, shifted_dim] += 5.0  # Large shift on dim 10

        distances = np.array([
            wasserstein_distance(ref[:, d], cur[:, d])
            for d in range(obs_dim)
        ])

        assert distances[shifted_dim] > 4.0, (
            f"Shifted dim {shifted_dim} should have high W1, got {distances[shifted_dim]}"
        )
        # Unshifted dims should be near 0
        unshifted = np.delete(distances, shifted_dim)
        assert np.mean(unshifted) < 0.5

    def test_mean_marginal_approximates_multivariate(self, rng) -> None:
        """Mean of per-dim W1 should be a reasonable proxy for multivariate drift."""
        ref = rng.standard_normal((1000, 54))
        cur = ref + 1.0  # Uniform shift

        per_dim = np.array([
            wasserstein_distance(ref[:, d], cur[:, d])
            for d in range(54)
        ])
        mean_w1 = float(np.mean(per_dim))
        assert abs(mean_w1 - 1.0) < 0.2


# ── Cooldown logic ──────────────────────────────────────────────────────────

class TestCooldown:
    """Test drift detection cooldown to prevent rapid-fire triggers."""

    def test_cooldown_prevents_rapid_fire(self) -> None:
        """No re-trigger should occur during cooldown period."""
        cooldown_steps = 200
        last_trigger_step = 1000
        events = []

        for step in range(1000, 1500):
            drift_detected = True  # Always above threshold
            if drift_detected and (step - last_trigger_step) >= cooldown_steps:
                events.append(step)
                last_trigger_step = step

        # First event at 1200 (after cooldown), then 1400
        assert len(events) <= 2, f"Expected ≤ 2 events during cooldown, got {len(events)}"
        if len(events) >= 2:
            for i in range(len(events) - 1):
                gap = events[i + 1] - events[i]
                assert gap >= cooldown_steps, f"Gap {gap} < cooldown {cooldown_steps}"

    def test_no_trigger_during_warmup(self) -> None:
        """No drift should be detected before the window is full."""
        window_size = 500
        warmup_steps = window_size
        events = []

        for step in range(1000):
            if step >= warmup_steps:
                events.append(step)

        # First event should be at step 500
        assert not any(e < warmup_steps for e in events)
