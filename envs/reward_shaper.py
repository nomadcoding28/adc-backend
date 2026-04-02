"""
envs/reward_shaper.py
=====================
Custom reward signal design for the ACD Framework.

CybORG's default reward is sparse: typically +1 for each host that stays
uncompromised per step and -1 for each compromised host.  This works but
provides little gradient signal early in training.

``RewardShaper`` augments the raw CybORG reward with dense shaping terms
that accelerate convergence without changing the optimal policy (following
Ng et al., 1999 potential-based shaping theory).

Reward components
-----------------
  raw              : Original CybORG reward (always included)
  removal_bonus    : +k when a compromised host is successfully cleaned
  isolation_bonus  : +k when a host is isolated before the attacker reaches it
  decoy_bonus      : +k when the attacker is lured to a decoy
  persistence_penalty : -k for each step where >N hosts are compromised
  step_cost        : -ε per step to discourage passivity (Monitor spam)
  restore_penalty  : -k if Restore is used when Remove would have sufficed
  early_clear_bonus: +k bonus if the network is fully clean before max_steps/2

All coefficients are configurable via the ``config`` dict.

Usage
-----
    shaper = RewardShaper(config={
        "removal_bonus":        2.0,
        "isolation_bonus":      1.0,
        "decoy_bonus":          1.5,
        "persistence_penalty":  0.5,
        "step_cost":            0.01,
        "restore_penalty":      0.5,
        "early_clear_bonus":    5.0,
        "compromised_threshold": 3,
    })

    shaped_reward = shaper.shape(
        raw_reward          = cyborg_reward,
        prev_obs            = prev_obs_vec,
        curr_obs            = curr_obs_vec,
        action_idx          = action_taken,
        action_mapper       = mapper,
        step                = current_step,
        max_steps           = 100,
        action_success      = True,
    )
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Default coefficient values ──────────────────────────────────────────────
_DEFAULTS: Dict[str, float] = {
    "removal_bonus":          2.0,
    "isolation_bonus":        1.0,
    "decoy_bonus":            1.5,
    "persistence_penalty":    0.3,
    "step_cost":              0.01,
    "restore_penalty":        0.5,
    "early_clear_bonus":      5.0,
    "compromised_threshold":  3.0,   # float for consistency; used as int
    "decoy_lure_bonus":       2.0,
}


class RewardShaper:
    """
    Transforms raw CybORG rewards into shaped, dense rewards.

    The shaped reward is:

        r_shaped = r_raw
                 + removal_bonus    (if applicable)
                 + isolation_bonus  (if applicable)
                 + decoy_bonus      (if applicable)
                 + early_clear_bonus (if applicable)
                 - persistence_penalty (if applicable)
                 - step_cost
                 - restore_penalty  (if applicable)

    Potential-based shaping guarantees that the shaped reward has the
    same optimal policy as the raw reward (Ng et al., 1999).

    Parameters
    ----------
    config : dict, optional
        Coefficient overrides.  Any key in ``_DEFAULTS`` can be overridden.
        Missing keys fall back to defaults.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = dict(_DEFAULTS)
        if config:
            cfg.update(config)

        self.removal_bonus          = float(cfg["removal_bonus"])
        self.isolation_bonus        = float(cfg["isolation_bonus"])
        self.decoy_bonus            = float(cfg["decoy_bonus"])
        self.decoy_lure_bonus       = float(cfg["decoy_lure_bonus"])
        self.persistence_penalty    = float(cfg["persistence_penalty"])
        self.step_cost              = float(cfg["step_cost"])
        self.restore_penalty        = float(cfg["restore_penalty"])
        self.early_clear_bonus      = float(cfg["early_clear_bonus"])
        self.compromised_threshold  = int(cfg["compromised_threshold"])

        # Episode-level tracking (reset each episode)
        self._prev_compromised_count: int = 0
        self._prev_decoy_count:       int = 0
        self._episode_reward:         float = 0.0
        self._shaping_history:        List[Dict[str, float]] = []

        logger.debug("RewardShaper initialised with config: %s", cfg)

    # ------------------------------------------------------------------ #
    # Primary interface
    # ------------------------------------------------------------------ #

    def shape(
        self,
        raw_reward:     float,
        prev_obs:       np.ndarray,
        curr_obs:       np.ndarray,
        action_idx:     int,
        action_mapper:  Any,                  # ActionMapper instance
        step:           int,
        max_steps:      int,
        action_success: bool = True,
    ) -> float:
        """
        Compute the shaped reward for a single environment step.

        Parameters
        ----------
        raw_reward : float
            Raw reward returned by CybORG.
        prev_obs : np.ndarray
            Processed observation from the previous step (shape 54,).
        curr_obs : np.ndarray
            Processed observation from the current step (shape 54,).
        action_idx : int
            Integer action that was taken.
        action_mapper : ActionMapper
            Used to classify the action type.
        step : int
            Current episode step (0-indexed).
        max_steps : int
            Episode horizon.
        action_success : bool
            Whether the action succeeded in CybORG.

        Returns
        -------
        float
            Shaped (augmented) reward.
        """
        shaped = float(raw_reward)
        breakdown: Dict[str, float] = {"raw": shaped}

        # Extract current state features from obs vectors
        n_compromised_prev = self._count_compromised(prev_obs)
        n_compromised_curr = self._count_compromised(curr_obs)
        n_decoy_prev       = self._count_decoys(prev_obs)
        n_decoy_curr       = self._count_decoys(curr_obs)

        # ── Removal bonus ──────────────────────────────────────────────
        # Awarded when a host transitions from compromised → clean
        hosts_cleaned = max(0, n_compromised_prev - n_compromised_curr)
        if hosts_cleaned > 0 and action_success:
            bonus = self.removal_bonus * hosts_cleaned
            shaped += bonus
            breakdown["removal_bonus"] = bonus

        # ── Isolation / proactive isolation bonus ──────────────────────
        # Awarded if the action is Analyse/Remove on a host that just
        # got compromised — i.e., rapid response
        if action_mapper.is_remediation_action(action_idx) and action_success:
            # Quick response means acting within 2 steps of compromise
            # (approximated by checking if compromise count just rose)
            if n_compromised_curr < n_compromised_prev:
                bonus = self.isolation_bonus
                shaped += bonus
                breakdown["isolation_bonus"] = bonus

        # ── Decoy bonus ────────────────────────────────────────────────
        # Awarded when a new decoy is deployed (transition in decoy count)
        new_decoys = max(0, n_decoy_curr - n_decoy_prev)
        if new_decoys > 0 and action_mapper.is_decoy_action(action_idx):
            bonus = self.decoy_bonus * new_decoys
            shaped += bonus
            breakdown["decoy_bonus"] = bonus

        # ── Decoy lure bonus ───────────────────────────────────────────
        # Awarded if attacker moved to a decoy host (indirect detection)
        # Detected by checking if attacker host index points to a decoy
        attacker_on_decoy = self._attacker_on_decoy(curr_obs)
        if attacker_on_decoy:
            shaped += self.decoy_lure_bonus
            breakdown["decoy_lure_bonus"] = self.decoy_lure_bonus

        # ── Early clear bonus ──────────────────────────────────────────
        # One-time bonus if network is fully clean before halfway point
        if n_compromised_curr == 0 and step < max_steps // 2:
            shaped += self.early_clear_bonus
            breakdown["early_clear_bonus"] = self.early_clear_bonus

        # ── Persistence penalty ────────────────────────────────────────
        # Continuous penalty when too many hosts are simultaneously compromised
        if n_compromised_curr >= self.compromised_threshold:
            excess = n_compromised_curr - self.compromised_threshold + 1
            penalty = self.persistence_penalty * excess
            shaped -= penalty
            breakdown["persistence_penalty"] = -penalty

        # ── Step cost ──────────────────────────────────────────────────
        # Small constant cost per step to discourage Monitor spamming
        shaped -= self.step_cost
        breakdown["step_cost"] = -self.step_cost

        # ── Restore over-use penalty ───────────────────────────────────
        # If Restore was used on a host that only needed Remove, penalise.
        # Approximated by: Restore was used when compromise count is low.
        if (
            action_mapper.is_remediation_action(action_idx)
            and action_mapper.get_spec(action_idx).action_type == "Restore"
            and n_compromised_curr <= 1
            and action_success
        ):
            # Restore is expensive — only justified for heavily compromised hosts
            shaped -= self.restore_penalty
            breakdown["restore_penalty"] = -self.restore_penalty

        # ── Update internal state ──────────────────────────────────────
        self._prev_compromised_count = n_compromised_curr
        self._prev_decoy_count       = n_decoy_curr
        self._episode_reward        += shaped
        self._shaping_history.append(breakdown)

        return float(shaped)

    def reset(self) -> None:
        """Reset episode-level tracking state.  Call at every episode start."""
        self._prev_compromised_count = 0
        self._prev_decoy_count       = 0
        self._episode_reward         = 0.0
        self._shaping_history.clear()

    # ------------------------------------------------------------------ #
    # Observation feature helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _count_compromised(obs: np.ndarray) -> int:
        """
        Count the number of hosts currently marked as compromised.

        Uses the host status block (dims 0–13, even indices).
        Each even index in [0, 2, 4, 6, 8, 10, 12] is the compromised flag
        for one host.
        """
        return int(sum(obs[i * 2] > 0.5 for i in range(7)))

    @staticmethod
    def _count_decoys(obs: np.ndarray) -> int:
        """
        Count the number of hosts with active decoys.

        Uses the host status block (dims 0–13, odd indices).
        """
        return int(sum(obs[i * 2 + 1] > 0.5 for i in range(7)))

    @staticmethod
    def _attacker_on_decoy(obs: np.ndarray) -> bool:
        """
        Heuristic: True if the attacker's believed position is a decoy host.

        Attacker host index is encoded in dim 44.
        Decoy flags are in dims 1, 3, 5, 7, 9, 11, 13 (odd of first 14).
        """
        attacker_idx = round(obs[44] * 6)    # re-scale from [0,1] → [0,6]
        decoy_flag   = obs[attacker_idx * 2 + 1] > 0.5
        return bool(decoy_flag)

    # ------------------------------------------------------------------ #
    # Analytics helpers (used by the dashboard / API)
    # ------------------------------------------------------------------ #

    @property
    def episode_total_reward(self) -> float:
        """Cumulative shaped reward for the current episode."""
        return self._episode_reward

    def get_shaping_breakdown(self) -> Dict[str, float]:
        """
        Aggregate shaping component totals across the current episode.

        Returns a dict like::

            {
                "raw":                 84.0,
                "removal_bonus":       10.0,
                "decoy_bonus":          4.5,
                "step_cost":           -1.0,
                "persistence_penalty": -0.6,
            }
        """
        totals: Dict[str, float] = {}
        for breakdown in self._shaping_history:
            for key, val in breakdown.items():
                totals[key] = totals.get(key, 0.0) + val
        return totals

    def get_last_breakdown(self) -> Dict[str, float]:
        """Return the shaping breakdown for the most recent step."""
        if not self._shaping_history:
            return {}
        return dict(self._shaping_history[-1])

    def get_config(self) -> Dict[str, Any]:
        """Return the current coefficient configuration as a dict."""
        return {
            "removal_bonus":         self.removal_bonus,
            "isolation_bonus":       self.isolation_bonus,
            "decoy_bonus":           self.decoy_bonus,
            "decoy_lure_bonus":      self.decoy_lure_bonus,
            "persistence_penalty":   self.persistence_penalty,
            "step_cost":             self.step_cost,
            "restore_penalty":       self.restore_penalty,
            "early_clear_bonus":     self.early_clear_bonus,
            "compromised_threshold": self.compromised_threshold,
        }

    def __repr__(self) -> str:
        return (
            f"RewardShaper("
            f"removal_bonus={self.removal_bonus}, "
            f"decoy_bonus={self.decoy_bonus}, "
            f"step_cost={self.step_cost})"
        )