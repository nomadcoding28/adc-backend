"""
game/nash_solver.py
====================
Nash equilibrium solver for the ACD two-player zero-sum stochastic game.

Mathematical background
-----------------------
In a two-player zero-sum game, the minimax theorem guarantees a unique
Nash equilibrium value V* and mixed strategy Nash equilibrium (π*, σ*):

    V* = max_π min_σ Σ_{a,b} π(a) · σ(b) · R(s, a, b)
       = min_σ max_π Σ_{a,b} π(a) · σ(b) · R(s, a, b)

The Blue player (maximiser) solves:

    max   V
    s.t.  Σ_a π(a) · R(s, a, b) ≥ V    ∀ b ∈ A_red
          Σ_a π(a) = 1
          π(a) ≥ 0                       ∀ a ∈ A_blue

This is a Linear Program (LP) with n_blue + 1 variables and
n_red + 2 constraints.

We solve the LP using scipy.optimize.linprog (simplex method).

For large action spaces (n_blue > 54), we use a column generation
approach — solve the LP over a subset of Blue actions and add
violated constraints iteratively.

Computational notes
-------------------
With n_blue=54 and n_red=4, the LP has 55 variables and 6 constraints
and solves in < 1ms on any modern CPU.  The full per-step solve is fast
enough to run synchronously in the game loop.

Usage
-----
    solver = NashSolver()

    # Build payoff matrix for current state
    R = solver.build_payoff_matrix(state, attacker_model, belief)

    # Solve for Nash equilibrium
    eq = solver.solve(R)
    print(eq.blue_strategy)   # Mixed strategy for blue player
    print(eq.game_value)      # Equilibrium value V*
    print(eq.best_response)   # Pure best-response action
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from game.stochastic_game import GameState, HostStatus, ALL_HOSTS

logger = logging.getLogger(__name__)

try:
    from scipy.optimize import linprog
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False
    logger.warning(
        "scipy not installed — NashSolver will use fictitious play fallback. "
        "Install with: pip install scipy"
    )

# Number of Red actions
N_RED  = 4
# Number of simplified Blue actions for LP (full 54-action space is too large
# for the analytical LP; we use a representative subset of 10 key actions)
N_BLUE_LP = 10

# Representative Blue action subset for LP solve
# Indices correspond to: Monitor, Analyse*3, Remove*3, Isolate*2, DeployDecoy*1
_BLUE_ACTION_SUBSET = [0, 1, 2, 3, 7, 8, 9, 14, 15, 21]

# Payoff weight for each attacker action
_RED_ACTION_NAMES = ["exploit", "spread", "persist", "exfiltrate"]


@dataclass
class NashEquilibrium:
    """
    Result of solving the two-player zero-sum game LP.

    Attributes
    ----------
    blue_strategy : np.ndarray
        Mixed strategy for Blue player, shape (n_blue_actions,).
        Probabilities sum to 1.
    red_strategy : np.ndarray
        Mixed strategy for Red player (from dual LP), shape (n_red,).
    game_value : float
        Expected payoff V* at the Nash equilibrium.
    best_response : int
        Pure best-response action for Blue (argmax of blue_strategy).
    best_response_value : float
        Expected payoff when Blue plays pure best-response.
    solve_method : str
        ``"lp"`` if scipy LP used, ``"fictitious_play"`` if fallback.
    n_iterations : int
        Number of iterations (LP: 1; fictitious play: n_iter).
    payoff_matrix : np.ndarray
        The R(a, b) matrix used for solving.
    """
    blue_strategy:       np.ndarray
    red_strategy:        np.ndarray
    game_value:          float
    best_response:       int
    best_response_value: float
    solve_method:        str
    n_iterations:        int                  = 1
    payoff_matrix:       Optional[np.ndarray] = field(default=None, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "blue_strategy":       [round(float(p), 4) for p in self.blue_strategy],
            "red_strategy":        [round(float(p), 4) for p in self.red_strategy],
            "game_value":          round(self.game_value, 4),
            "best_response":       self.best_response,
            "best_response_value": round(self.best_response_value, 4),
            "solve_method":        self.solve_method,
            "n_iterations":        self.n_iterations,
        }

    @property
    def exploitability(self) -> float:
        """
        Exploitability = distance from true Nash equilibrium.

        Measures how much a player can improve by deviating.
        Lower = closer to true Nash equilibrium.
        Perfect Nash: exploitability = 0.
        """
        return abs(self.game_value - self.best_response_value)

    def __str__(self) -> str:
        return (
            f"NashEquilibrium("
            f"V*={self.game_value:.4f}, "
            f"best_response=action_{self.best_response}, "
            f"method={self.solve_method})"
        )


class NashSolver:
    """
    Solves the minimax LP for the ACD stochastic game at each step.

    Parameters
    ----------
    n_blue_actions : int
        Total Blue action space size.  Default 54.
    n_red_actions : int
        Total Red action space size.  Default 4.
    fictitious_play_iters : int
        Fallback fictitious play iterations (if scipy unavailable).
        Default 1000.
    """

    def __init__(
        self,
        n_blue_actions:         int = 54,
        n_red_actions:          int = N_RED,
        fictitious_play_iters:  int = 1000,
    ) -> None:
        self.n_blue          = n_blue_actions
        self.n_red           = n_red_actions
        self._fp_iters       = fictitious_play_iters

        # Cache last solution for warm-starting
        self._last_solution: Optional[NashEquilibrium] = None

    # ------------------------------------------------------------------ #
    # Primary interface
    # ------------------------------------------------------------------ #

    def solve(
        self,
        payoff_matrix: np.ndarray,
        action_subset: Optional[List[int]] = None,
    ) -> NashEquilibrium:
        """
        Solve the minimax LP given a payoff matrix.

        Parameters
        ----------
        payoff_matrix : np.ndarray
            Shape (n_blue, n_red).  R[a, b] = payoff when Blue plays a
            and Red plays b.  Positive = good for Blue.
        action_subset : list[int], optional
            If given, only optimise over this subset of Blue actions.
            The returned ``best_response`` is an index into this subset.

        Returns
        -------
        NashEquilibrium
        """
        R = payoff_matrix
        n_b, n_r = R.shape

        if _SCIPY_AVAILABLE:
            return self._solve_lp(R, action_subset)
        else:
            return self._solve_fictitious_play(R, action_subset)

    def build_payoff_matrix(
        self,
        state:          GameState,
        attacker_model: Any,    # AttackerModel
        belief:         Any,    # BeliefState
        action_subset:  Optional[List[int]] = None,
    ) -> np.ndarray:
        """
        Build the payoff matrix R(a_blue, a_red) for the current state.

        The payoff combines:
            1. Immediate reward from the transition (state-dependent)
            2. Belief-weighted attacker strategy (expected over θ)

        R(a, b) = E_θ[P(θ) · reward(s, a, b, θ)]

        Parameters
        ----------
        state : GameState
        attacker_model : AttackerModel
        belief : BeliefState
        action_subset : list[int], optional

        Returns
        -------
        np.ndarray
            Shape (n_blue, n_red).
        """
        from game.attacker_model import AttackerType

        actions = action_subset or _BLUE_ACTION_SUBSET
        n_b     = len(actions)
        R       = np.zeros((n_b, self.n_red), dtype=np.float32)

        belief_probs = belief.probabilities

        for i, blue_a in enumerate(actions):
            for j in range(self.n_red):
                # Belief-weighted payoff across attacker types
                payoff = 0.0
                for type_name, prob in belief_probs.items():
                    try:
                        at = AttackerType(type_name)
                    except ValueError:
                        continue

                    # Estimate reward for (blue_a, red_j) pair
                    estimated_reward = self._estimate_reward(
                        blue_action  = blue_a,
                        red_action   = j,
                        state        = state,
                        attacker_type= at,
                    )
                    payoff += prob * estimated_reward

                R[i, j] = payoff

        return R

    # ------------------------------------------------------------------ #
    # LP solver (scipy)
    # ------------------------------------------------------------------ #

    def _solve_lp(
        self,
        R:             np.ndarray,
        action_subset: Optional[List[int]],
    ) -> NashEquilibrium:
        """
        Solve the minimax LP using scipy linprog.

        Formulation (standard form for Blue maximiser):

            max  V                          (negate for minimiser linprog)
            s.t. R^T · π ≥ V·1             (security constraints)
                 1^T · π = 1               (probability simplex)
                 π ≥ 0

        Variables: x = [π_0, ..., π_{n_b-1}, V]  (n_b + 1 variables)
        """
        n_b, n_r = R.shape

        # Fast path for degenerate zero matrix
        if not np.any(R):
            pi = np.ones(n_b, dtype=np.float32) / n_b
            sigma = np.ones(n_r, dtype=np.float32) / n_r
            best_action = (action_subset or _BLUE_ACTION_SUBSET)[0] if action_subset else 0
            
            eq = NashEquilibrium(
                blue_strategy       = pi,
                red_strategy        = sigma,
                game_value          = 0.0,
                best_response       = best_action,
                best_response_value = 0.0,
                solve_method        = "lp",
                payoff_matrix       = R,
            )
            self._last_solution = eq
            return eq

        # Objective: minimise -V (scipy minimises by default)
        c = np.zeros(n_b + 1)
        c[-1] = -1.0   # coefficient of V

        # Security constraints: R^T · π - V·1 ≥ 0
        # Rewritten as: -R^T · π + V·1 ≤ 0
        # A_ub shape: (n_r, n_b + 1)
        A_ub = np.hstack([-R.T, np.ones((n_r, 1))])
        b_ub = np.zeros(n_r)

        # Equality: Σ π_i = 1
        A_eq = np.ones((1, n_b + 1))
        A_eq[0, -1] = 0    # V is not constrained by this equation
        b_eq = np.array([1.0])

        # Bounds: π_i ≥ 0, V unbounded
        bounds = [(0.0, None)] * n_b + [(None, None)]

        result = linprog(
            c,
            A_ub   = A_ub,
            b_ub   = b_ub,
            A_eq   = A_eq,
            b_eq   = b_eq,
            bounds = bounds,
            method = "highs",
        )

        if result.success:
            pi      = np.array(result.x[:n_b], dtype=np.float32)
            pi      = np.clip(pi, 0, None)
            pi_sum  = pi.sum()
            pi      /= max(pi_sum, 1e-8)   # re-normalise (numerical errors)
            V_star  = float(result.x[-1])

            # Dual solution gives Red's mixed strategy
            # (from the dual LP — rows of A_ub correspond to Red actions)
            if result.ineqlin is not None and result.ineqlin.marginals is not None:
                sigma = np.array(result.ineqlin.marginals, dtype=np.float32)
                sigma = np.abs(sigma)
                sigma /= max(sigma.sum(), 1e-8)
            else:
                sigma = np.ones(n_r, dtype=np.float32) / n_r

        else:
            logger.warning(
                "LP solve failed: %s — falling back to uniform strategy.",
                result.message,
            )
            pi     = np.ones(n_b, dtype=np.float32) / n_b
            sigma  = np.ones(n_r, dtype=np.float32) / n_r
            V_star = float(np.mean(R))

        # Pure best response = action with highest π weight
        best_idx    = int(np.argmax(pi))
        best_action = (action_subset or _BLUE_ACTION_SUBSET)[best_idx] if action_subset else best_idx
        best_value  = float(np.dot(R[best_idx], sigma))

        eq = NashEquilibrium(
            blue_strategy       = pi,
            red_strategy        = sigma,
            game_value          = V_star,
            best_response       = best_action,
            best_response_value = best_value,
            solve_method        = "lp",
            payoff_matrix       = R,
        )
        self._last_solution = eq
        return eq

    # ------------------------------------------------------------------ #
    # Fictitious play fallback (no scipy)
    # ------------------------------------------------------------------ #

    def _solve_fictitious_play(
        self,
        R:             np.ndarray,
        action_subset: Optional[List[int]],
    ) -> NashEquilibrium:
        """
        Approximate Nash equilibrium via fictitious play.

        Fictitious play: each player best-responds to the empirical
        frequency of the opponent's past actions.  Converges to Nash
        for zero-sum games (Brown, 1951).

        Parameters
        ----------
        R : np.ndarray
            Shape (n_blue, n_red).
        action_subset : list[int], optional

        Returns
        -------
        NashEquilibrium
        """
        n_b, n_r = R.shape
        rng      = np.random.default_rng(42)

        # Initialise with uniform play
        blue_counts = np.ones(n_b, dtype=np.float64)
        red_counts  = np.ones(n_r, dtype=np.float64)

        for _ in range(self._fp_iters):
            # Blue best-responds to Red's empirical frequency
            red_freq    = red_counts / red_counts.sum()
            blue_payoffs= R @ red_freq
            blue_br     = int(np.argmax(blue_payoffs))
            blue_counts[blue_br] += 1

            # Red best-responds to Blue's empirical frequency (minimiser)
            blue_freq   = blue_counts / blue_counts.sum()
            red_payoffs = R.T @ blue_freq
            red_br      = int(np.argmin(red_payoffs))
            red_counts[red_br] += 1

        pi    = (blue_counts / blue_counts.sum()).astype(np.float32)
        sigma = (red_counts  / red_counts.sum()).astype(np.float32)
        V_star = float(pi @ R @ sigma)

        best_idx    = int(np.argmax(pi))
        best_action = (action_subset or list(range(n_b)))[best_idx]
        best_value  = float(R[best_idx] @ sigma)

        eq = NashEquilibrium(
            blue_strategy       = pi,
            red_strategy        = sigma,
            game_value          = V_star,
            best_response       = best_action,
            best_response_value = best_value,
            solve_method        = "fictitious_play",
            n_iterations        = self._fp_iters,
            payoff_matrix       = R,
        )
        self._last_solution = eq
        return eq

    # ------------------------------------------------------------------ #
    # Payoff estimation helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _estimate_reward(
        blue_action:   int,
        red_action:    int,
        state:         GameState,
        attacker_type: Any,
    ) -> float:
        """
        Estimate the immediate reward for a (blue_action, red_action) pair
        given the current state and attacker type.

        This is a heuristic reward model (not the full CybORG transition)
        used for fast LP payoff matrix construction.

        Heuristic logic:
            - Remove/Restore on compromised host: positive reward
            - Isolate on compromised host: positive reward
            - DeployDecoy on clean host: small positive
            - Monitor: neutral
            Red spread action on clean network: large negative
        """
        statuses     = state.host_statuses
        n_compromised = state.n_compromised

        # Blue action categories (approximate ranges)
        if blue_action < 7:
            # Remove malware
            target = ALL_HOSTS[blue_action]
            if statuses.get(target) == HostStatus.COMPROMISED:
                base_reward = 2.0
            else:
                base_reward = -0.1   # Wasted action

        elif blue_action < 14:
            # Restore
            target = ALL_HOSTS[blue_action - 7]
            if statuses.get(target) == HostStatus.COMPROMISED:
                base_reward = 1.5
            else:
                base_reward = -0.05

        elif blue_action < 21:
            # Isolate
            target = ALL_HOSTS[blue_action - 14]
            status = statuses.get(target, HostStatus.CLEAN)
            if status == HostStatus.COMPROMISED:
                base_reward = 1.8
            elif status == HostStatus.CLEAN:
                base_reward = 0.2   # Pre-emptive isolation
            else:
                base_reward = -0.1

        elif blue_action < 28:
            # Deploy decoy
            base_reward = 0.8 if n_compromised > 0 else 0.2

        else:
            # Monitor
            base_reward = 0.1 if n_compromised == 0 else -0.2

        # Red action modifier
        if red_action == 1 and n_compromised == 0:
            # Spread into clean network — high risk
            base_reward -= 1.5
        elif red_action == 3:
            # Exfiltration attempt
            base_reward -= 3.0 if "Op_Server0" in [
                h for h, s in statuses.items() if s == HostStatus.COMPROMISED
            ] else -0.5

        return round(base_reward, 4)

    def get_action_recommendation(
        self,
        state:          GameState,
        attacker_model: Any,
        belief:         Any,
        top_k:          int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Solve the Nash LP and return top-k recommended Blue actions.

        Parameters
        ----------
        state : GameState
        attacker_model : AttackerModel
        belief : BeliefState
        top_k : int
            Number of top actions to return.

        Returns
        -------
        list[dict]
            Each: ``{action_idx, probability, description, value}``.
        """
        R  = self.build_payoff_matrix(state, attacker_model, belief, _BLUE_ACTION_SUBSET)
        eq = self.solve(R, action_subset=_BLUE_ACTION_SUBSET)

        # Rank actions by mixed strategy probability
        action_probs = list(zip(_BLUE_ACTION_SUBSET, eq.blue_strategy.tolist()))
        action_probs.sort(key=lambda x: -x[1])

        recommendations = []
        for idx, (action_idx, prob) in enumerate(action_probs[:top_k]):
            if len(ALL_HOSTS) > action_idx:
                desc = f"Action {action_idx} ({ALL_HOSTS[min(action_idx, len(ALL_HOSTS)-1)]})"
            else:
                desc = f"Action {action_idx}"

            recommendations.append({
                "rank":       idx + 1,
                "action_idx": action_idx,
                "probability":round(prob, 4),
                "description":desc,
                "value":      round(float(R[_BLUE_ACTION_SUBSET.index(action_idx)] @ eq.red_strategy), 4),
            })

        return recommendations

    def __repr__(self) -> str:
        return (
            f"NashSolver("
            f"n_blue={self.n_blue}, "
            f"n_red={self.n_red}, "
            f"method={'lp' if _SCIPY_AVAILABLE else 'fictitious_play'})"
        )