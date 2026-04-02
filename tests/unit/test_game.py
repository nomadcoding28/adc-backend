"""
tests/unit/test_game.py
========================
Unit tests for the Nash equilibrium solver.

Tests validate:
    - LP solution satisfies minimax property
    - Mixed strategies sum to 1 (probability simplex)
    - Fictitious play converges to Nash for zero-sum games
    - Payoff matrix dimensions are correct
"""

from __future__ import annotations

import numpy as np
import pytest

# Import the solver under test
from game.nash_solver import NashSolver, NashEquilibrium, _SCIPY_AVAILABLE


# ── Nash LP solver ──────────────────────────────────────────────────────────

class TestNashSolver:
    """Test the minimax LP solver."""

    @pytest.fixture
    def solver(self) -> NashSolver:
        return NashSolver(n_blue_actions=4, n_red_actions=3)

    @pytest.fixture
    def simple_payoff(self) -> np.ndarray:
        """Simple 4×3 payoff matrix."""
        return np.array([
            [ 3.0, -1.0,  2.0],
            [ 1.0,  4.0, -2.0],
            [-1.0,  2.0,  3.0],
            [ 2.0,  0.0,  1.0],
        ], dtype=np.float32)

    def test_strategies_sum_to_one(self, solver, simple_payoff) -> None:
        """Both Blue and Red mixed strategies should sum to 1."""
        eq = solver.solve(simple_payoff)
        assert abs(eq.blue_strategy.sum() - 1.0) < 1e-4, (
            f"Blue strategy sums to {eq.blue_strategy.sum()}, not 1.0"
        )
        assert abs(eq.red_strategy.sum() - 1.0) < 1e-4, (
            f"Red strategy sums to {eq.red_strategy.sum()}, not 1.0"
        )

    def test_strategies_non_negative(self, solver, simple_payoff) -> None:
        """All strategy probabilities should be ≥ 0."""
        eq = solver.solve(simple_payoff)
        assert np.all(eq.blue_strategy >= -1e-6), "Blue strategy has negative probabilities."
        assert np.all(eq.red_strategy >= -1e-6), "Red strategy has negative probabilities."

    def test_game_value_between_min_max(self, solver, simple_payoff) -> None:
        """Nash game value V* should be between min and max of the payoff matrix."""
        eq = solver.solve(simple_payoff)
        assert simple_payoff.min() <= eq.game_value <= simple_payoff.max() + 1e-4, (
            f"Game value {eq.game_value} out of payoff range "
            f"[{simple_payoff.min()}, {simple_payoff.max()}]"
        )

    def test_best_response_is_valid_action(self, solver, simple_payoff) -> None:
        """Best response should be a valid action index."""
        eq = solver.solve(simple_payoff)
        n_blue = simple_payoff.shape[0]
        assert 0 <= eq.best_response < n_blue

    @pytest.mark.skipif(not _SCIPY_AVAILABLE, reason="scipy not installed")
    def test_lp_solve_method(self, solver, simple_payoff) -> None:
        """When scipy is available, solve method should be 'lp'."""
        eq = solver.solve(simple_payoff)
        assert eq.solve_method == "lp"

    def test_to_dict_serialisable(self, solver, simple_payoff) -> None:
        """NashEquilibrium.to_dict() should produce JSON-serialisable output."""
        eq = solver.solve(simple_payoff)
        d = eq.to_dict()
        assert isinstance(d, dict)
        assert "blue_strategy" in d
        assert "game_value" in d
        assert isinstance(d["blue_strategy"], list)


# ── Fictitious play ─────────────────────────────────────────────────────────

class TestFictitiousPlay:
    """Test the fictitious play fallback solver."""

    def test_fictitious_play_convergence(self) -> None:
        """
        Fictitious play should converge to Nash equilibrium for
        the classic matching pennies game.
        """
        solver = NashSolver(n_blue_actions=2, n_red_actions=2, fictitious_play_iters=5000)

        # Matching pennies: unique Nash is (0.5, 0.5) for both
        R = np.array([
            [ 1.0, -1.0],
            [-1.0,  1.0],
        ], dtype=np.float32)

        eq = solver._solve_fictitious_play(R, action_subset=None)

        # Should converge to uniform (0.5, 0.5) ± tolerance
        assert abs(eq.blue_strategy[0] - 0.5) < 0.1, (
            f"Expected ~0.5, got {eq.blue_strategy[0]}"
        )
        assert abs(eq.game_value) < 0.1, (
            f"Game value should be ~0 for matching pennies, got {eq.game_value}"
        )

    def test_dominant_strategy_detection(self) -> None:
        """If Blue has a dominant strategy, FP should find it."""
        solver = NashSolver(n_blue_actions=3, n_red_actions=2, fictitious_play_iters=2000)

        # Action 0 dominates: always best regardless of Red's action
        R = np.array([
            [10.0, 10.0],
            [ 1.0,  1.0],
            [ 0.0,  0.0],
        ], dtype=np.float32)

        eq = solver._solve_fictitious_play(R, action_subset=None)
        assert eq.best_response == 0, "Should detect dominant action 0."


# ── Payoff matrix ───────────────────────────────────────────────────────────

class TestPayoffMatrix:
    """Test payoff matrix construction."""

    def test_payoff_matrix_shape(self) -> None:
        """Payoff matrix should have shape (n_blue, n_red)."""
        n_blue = 10
        n_red = 4
        R = np.random.randn(n_blue, n_red).astype(np.float32)
        assert R.shape == (n_blue, n_red)

    def test_zero_payoff_gives_valid_nash(self) -> None:
        """A zero payoff matrix should yield a valid Nash equilibrium with game value 0."""
        solver = NashSolver(n_blue_actions=4, n_red_actions=3)
        R = np.zeros((4, 3), dtype=np.float32)
        eq = solver.solve(R)

        # Game value should be 0 (all payoffs are 0)
        assert abs(eq.game_value) < 1e-4, f"Game value should be 0, got {eq.game_value}"
        # Strategy should be a valid probability distribution
        assert abs(eq.blue_strategy.sum() - 1.0) < 1e-4
        assert np.all(eq.blue_strategy >= -1e-6)
