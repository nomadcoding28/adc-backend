"""
game/belief_updater.py
=======================
Bayesian belief updater — maintains P(θ | history) over attacker types.

Mathematical formulation
------------------------
Let θ ∈ {Random, TargetedAPT, Adaptive} be the attacker type.
Let o_t be the observation at step t (attacker's action + affected hosts).

The Bayesian posterior update is:

    P(θ | o_1:t) ∝ P(o_t | θ, s_t) · P(θ | o_1:t-1)

where:
    P(o_t | θ, s_t) = AttackerModel.likelihood_of_observation(o_t, θ, s_t)

In log space (for numerical stability):

    log P(θ | o_1:t) = log P(o_t | θ, s_t) + log P(θ | o_1:t-1) + const

Belief is normalised after each update so probabilities sum to 1.

The prior P(θ) is configurable — default uniform (1/3 each).

Usage
-----
    updater = BeliefUpdater(prior={
        "Random":      0.33,
        "TargetedAPT": 0.34,
        "Adaptive":    0.33,
    })

    # Update belief after observing attacker action
    belief = updater.update(
        observation    = {"red_action": 1, "hosts_affected": ["User1"]},
        state          = game_state,
        attacker_model = model,
    )

    print(belief.dominant_type)           # "TargetedAPT"
    print(belief.probabilities)           # {"Random": 0.12, ...}
    print(belief.entropy)                 # 0.72 bits
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from game.attacker_model import AttackerModel, AttackerType
from game.stochastic_game import GameState

logger = logging.getLogger(__name__)

# Minimum probability floor (avoids complete collapse to zero)
_PROB_FLOOR    = 1e-6
# Learning rate for exponential moving average of likelihoods
_EMA_ALPHA     = 0.1


@dataclass
class BeliefState:
    """
    Current Bayesian belief over attacker type.

    Attributes
    ----------
    probabilities : dict[str, float]
        P(θ | history) for each attacker type.
    log_probs : dict[str, float]
        Log-space belief (more numerically stable).
    step : int
        Episode step at which this belief was computed.
    last_observation : dict
        The observation that triggered the last update.
    update_count : int
        Total number of Bayesian updates applied so far.
    history : list[dict]
        Full belief history (one entry per update step).
    """
    probabilities:    Dict[str, float]
    log_probs:        Dict[str, float]
    step:             int                 = 0
    last_observation: Dict[str, Any]      = field(default_factory=dict)
    update_count:     int                 = 0
    history:          List[Dict[str, Any]] = field(default_factory=list)

    @property
    def dominant_type(self) -> str:
        """The attacker type with the highest posterior probability."""
        return max(self.probabilities, key=self.probabilities.get)

    @property
    def dominant_probability(self) -> float:
        """Posterior probability of the dominant type."""
        return self.probabilities.get(self.dominant_type, 0.0)

    @property
    def entropy(self) -> float:
        """
        Shannon entropy of the belief distribution (in bits).

        High entropy = uncertain about attacker type.
        Low entropy  = confident about attacker type.
        Max entropy (uniform over 3 types) ≈ 1.585 bits.
        """
        h = 0.0
        for p in self.probabilities.values():
            if p > 0:
                h -= p * math.log2(p)
        return round(h, 4)

    @property
    def is_confident(self) -> bool:
        """True if the dominant type exceeds the confidence threshold (60%)."""
        return self.dominant_probability >= 0.60

    def to_dict(self) -> Dict[str, Any]:
        return {
            "probabilities":       {k: round(v, 4) for k, v in self.probabilities.items()},
            "dominant_type":       self.dominant_type,
            "dominant_probability": round(self.dominant_probability, 4),
            "entropy":             self.entropy,
            "step":                self.step,
            "update_count":        self.update_count,
            "is_confident":        self.is_confident,
        }

    def __str__(self) -> str:
        probs_str = ", ".join(
            f"{k}={v:.2f}" for k, v in sorted(
                self.probabilities.items(), key=lambda x: -x[1]
            )
        )
        return f"BeliefState(step={self.step}, [{probs_str}], H={self.entropy:.3f})"


class BeliefUpdater:
    """
    Bayesian posterior updater for attacker type identification.

    Parameters
    ----------
    prior : dict[str, float], optional
        Prior probabilities over attacker types.
        Default: uniform (1/3 each).
    smoothing : float
        Laplace smoothing coefficient added to likelihoods.
        Prevents belief collapse.  Default 0.05.
    ema_alpha : float
        Exponential moving average factor for likelihood smoothing.
        Higher = faster adaptation to new evidence.  Default 0.1.
    max_history : int
        Maximum belief history entries to retain.  Default 200.
    """

    _TYPE_NAMES = [t.value for t in AttackerType.all_types()]

    def __init__(
        self,
        prior:       Optional[Dict[str, float]] = None,
        smoothing:   float = 0.05,
        ema_alpha:   float = _EMA_ALPHA,
        max_history: int   = 200,
    ) -> None:
        # Default: uniform prior
        if prior is None:
            n = len(self._TYPE_NAMES)
            prior = {t: 1.0 / n for t in self._TYPE_NAMES}

        self._validate_prior(prior)
        self.smoothing   = smoothing
        self.ema_alpha   = ema_alpha
        self.max_history = max_history

        # Initialise log-space belief from prior
        self._log_belief: Dict[str, float] = {
            t: math.log(max(p, _PROB_FLOOR))
            for t, p in prior.items()
        }
        self._prior = dict(prior)

        # Episode-level tracking
        self._update_count: int = 0
        self._history:      List[Dict[str, Any]] = []

        # Smoothed likelihood estimates (EMA) for tracking
        self._ema_likelihoods: Dict[str, float] = {
            t: 1.0 / len(self._TYPE_NAMES) for t in self._TYPE_NAMES
        }

        logger.debug("BeliefUpdater initialised — prior=%s", prior)

    # ------------------------------------------------------------------ #
    # Primary interface
    # ------------------------------------------------------------------ #

    def update(
        self,
        observation:    Dict[str, Any],
        state:          GameState,
        attacker_model: AttackerModel,
    ) -> BeliefState:
        """
        Apply a Bayesian update given a new observation.

        P(θ | o_1:t) ∝ P(o_t | θ, s_t) · P(θ | o_1:t-1)

        Parameters
        ----------
        observation : dict
            Keys: ``red_action`` (int), ``hosts_affected`` (list[str]),
            optionally ``attacker_position`` (str).
        state : GameState
            Current game state (used by attacker model for likelihood).
        attacker_model : AttackerModel
            The attacker model that computes P(o | θ, s).

        Returns
        -------
        BeliefState
            Updated belief distribution.
        """
        likelihoods: Dict[str, float] = {}

        for type_name in self._TYPE_NAMES:
            try:
                attacker_type = AttackerType(type_name)
            except ValueError:
                likelihoods[type_name] = 1.0 / len(self._TYPE_NAMES)
                continue

            # P(observation | θ, state)
            raw_likelihood = attacker_model.likelihood_of_observation(
                observation    = observation,
                attacker_type  = attacker_type,
                state          = state,
            )

            # Apply Laplace smoothing
            smoothed = raw_likelihood + self.smoothing

            # EMA smoothing for stability across steps
            self._ema_likelihoods[type_name] = (
                self.ema_alpha * smoothed
                + (1 - self.ema_alpha) * self._ema_likelihoods[type_name]
            )
            likelihoods[type_name] = self._ema_likelihoods[type_name]

        # Log-space Bayesian update
        for type_name, likelihood in likelihoods.items():
            self._log_belief[type_name] += math.log(max(likelihood, _PROB_FLOOR))

        # Normalise (log-sum-exp for numerical stability)
        self._normalise_log_belief()

        # Convert to probability space
        probs = {t: math.exp(lp) for t, lp in self._log_belief.items()}

        self._update_count += 1
        belief = BeliefState(
            probabilities    = probs,
            log_probs        = dict(self._log_belief),
            step             = state.step,
            last_observation = dict(observation),
            update_count     = self._update_count,
        )

        # Record history
        self._history.append(belief.to_dict())
        if len(self._history) > self.max_history:
            self._history.pop(0)

        logger.debug(
            "Belief updated — step=%d, dominant=%s (%.2f), H=%.3f",
            state.step,
            belief.dominant_type,
            belief.dominant_probability,
            belief.entropy,
        )

        return belief

    def update_batch(
        self,
        observations:   List[Dict[str, Any]],
        states:         List[GameState],
        attacker_model: AttackerModel,
    ) -> List[BeliefState]:
        """
        Apply sequential Bayesian updates for a list of observations.

        Parameters
        ----------
        observations : list[dict]
        states : list[GameState]
        attacker_model : AttackerModel

        Returns
        -------
        list[BeliefState]
            Belief state after each observation.
        """
        beliefs: List[BeliefState] = []
        for obs, state in zip(observations, states):
            belief = self.update(obs, state, attacker_model)
            beliefs.append(belief)
        return beliefs

    def get_current_belief(self) -> BeliefState:
        """
        Return the current belief state without applying a new update.

        Returns
        -------
        BeliefState
        """
        probs = {t: math.exp(lp) for t, lp in self._log_belief.items()}
        return BeliefState(
            probabilities = probs,
            log_probs     = dict(self._log_belief),
            step          = 0,
            update_count  = self._update_count,
            history       = list(self._history[-20:]),  # last 20 for API
        )

    # ------------------------------------------------------------------ #
    # Analysis helpers
    # ------------------------------------------------------------------ #

    def get_probability(self, attacker_type: str) -> float:
        """
        Return the current posterior probability for a single attacker type.

        Parameters
        ----------
        attacker_type : str
            e.g. ``"TargetedAPT"``.

        Returns
        -------
        float
        """
        lp = self._log_belief.get(attacker_type, math.log(_PROB_FLOOR))
        return math.exp(lp)

    def get_belief_history(
        self, last_n: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Return the full belief history as a list of dicts.

        Parameters
        ----------
        last_n : int, optional
            Return only the last N entries.

        Returns
        -------
        list[dict]
        """
        history = self._history
        if last_n is not None:
            history = history[-last_n:]
        return list(history)

    def belief_shift_detected(self, window: int = 10, threshold: float = 0.15) -> bool:
        """
        Detect if the dominant belief has shifted significantly over a window.

        Parameters
        ----------
        window : int
            Number of recent steps to compare.
        threshold : float
            Minimum probability shift to flag as a change.

        Returns
        -------
        bool
            True if the dominant type changed or dominant probability shifted
            by more than ``threshold``.
        """
        if len(self._history) < window:
            return False

        old = self._history[-window]
        new = self._history[-1]

        old_dominant = max(old["probabilities"], key=old["probabilities"].get)
        new_dominant = max(new["probabilities"], key=new["probabilities"].get)

        if old_dominant != new_dominant:
            return True

        old_p = old["probabilities"].get(old_dominant, 0.0)
        new_p = new["probabilities"].get(new_dominant, 0.0)

        return abs(new_p - old_p) >= threshold

    def compute_kullback_leibler(
        self, other: Dict[str, float]
    ) -> float:
        """
        Compute KL divergence from current belief to a reference distribution.

        KL(P_current || P_other) = Σ_θ P(θ) · log(P(θ) / Q(θ))

        Parameters
        ----------
        other : dict[str, float]
            Reference distribution over attacker types.

        Returns
        -------
        float
            KL divergence in nats.  0 = distributions are identical.
        """
        current = {t: math.exp(lp) for t, lp in self._log_belief.items()}
        kl = 0.0
        for t in self._TYPE_NAMES:
            p = max(current.get(t, _PROB_FLOOR), _PROB_FLOOR)
            q = max(other.get(t, _PROB_FLOOR), _PROB_FLOOR)
            kl += p * (math.log(p) - math.log(q))
        return round(kl, 6)

    def get_recommended_strategy(self) -> str:
        """
        Return a natural-language strategy recommendation based on belief.

        Returns
        -------
        str
        """
        belief = self.get_current_belief()
        dt     = belief.dominant_type
        dp     = belief.dominant_probability

        if not belief.is_confident:
            return (
                "Belief is uncertain (entropy={:.2f} bits). "
                "Use Analyse action to gather more information before committing "
                "to a defensive strategy.".format(belief.entropy)
            )

        if dt == AttackerType.RANDOM.value:
            return (
                f"Random attacker detected ({dp:.0%} confidence). "
                "Standard monitoring and decoy deployment is sufficient. "
                "Low priority — respond reactively."
            )

        elif dt == AttackerType.TARGETED_APT.value:
            return (
                f"Targeted APT detected ({dp:.0%} confidence). "
                "Pre-emptively isolate high-value hosts (Enterprise0, Op_Server0). "
                "Deploy decoys on intermediate subnet hosts to delay lateral movement. "
                "HIGH PRIORITY — act before persistence is established."
            )

        else:  # Adaptive
            return (
                f"Adaptive attacker detected ({dp:.0%} confidence). "
                "Randomise defensive actions to avoid predictable patterns. "
                "Mix isolation, removal, and decoy deployment unpredictably. "
                "Monitor for strategy shifts — this attacker responds to your TTPs."
            )

    # ------------------------------------------------------------------ #
    # Reset
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        """
        Reset belief to the prior distribution.

        Call at the start of each new episode.
        """
        self._log_belief = {
            t: math.log(max(p, _PROB_FLOOR))
            for t, p in self._prior.items()
        }
        self._update_count = 0
        self._history.clear()
        self._ema_likelihoods = {
            t: 1.0 / len(self._TYPE_NAMES) for t in self._TYPE_NAMES
        }
        logger.debug("BeliefUpdater reset to prior.")

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _normalise_log_belief(self) -> None:
        """
        Normalise log-space belief using the log-sum-exp trick.

        log_sum_exp(x) = max(x) + log(Σ exp(x_i - max(x)))
        """
        log_values = list(self._log_belief.values())
        max_log    = max(log_values)
        log_sum    = max_log + math.log(
            sum(math.exp(lv - max_log) for lv in log_values)
        )
        for t in self._log_belief:
            self._log_belief[t] -= log_sum

    @staticmethod
    def _validate_prior(prior: Dict[str, float]) -> None:
        """Check that the prior is a valid probability distribution."""
        if abs(sum(prior.values()) - 1.0) > 0.01:
            raise ValueError(
                f"Prior probabilities must sum to 1.0, "
                f"got {sum(prior.values()):.4f}."
            )
        if any(p < 0 for p in prior.values()):
            raise ValueError("Prior probabilities must be non-negative.")

    def __repr__(self) -> str:
        current = {t: round(math.exp(lp), 3) for t, lp in self._log_belief.items()}
        return (
            f"BeliefUpdater("
            f"belief={current}, "
            f"updates={self._update_count})"
        )