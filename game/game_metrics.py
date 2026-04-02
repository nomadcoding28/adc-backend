"""
game/game_metrics.py
=====================
Game-theoretic performance metrics for the ACD Framework.

Tracks and computes metrics specific to the stochastic game:
    - Value of the game (Nash equilibrium value V*)
    - Best response gap (exploitability)
    - Attacker type prediction accuracy
    - Belief entropy over time
    - Defender win rate
    - Kill-chain stage distribution
    - Prediction accuracy for attacker's next action

These metrics are logged to TensorBoard, served via the API /game routes,
and used to populate the Game Model panel in the dashboard.

Usage
-----
    metrics = GameMetrics()

    # Record after each game step
    metrics.record_step(
        step          = 4821,
        belief        = belief_state,
        nash_eq       = nash_equilibrium,
        game_state    = game_state,
        actual_red_action = 1,
        predicted_red_action = 1,
    )

    # Get current snapshot for API
    snapshot = metrics.get_snapshot()
    print(snapshot.attacker_prediction_accuracy)

    # Get full history for the dashboard chart
    history = metrics.get_belief_history()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from game.stochastic_game import GameState, KillChainStage
from game.belief_updater import BeliefState
from game.attacker_model import AttackerType

logger = logging.getLogger(__name__)

# Rolling window sizes
_WINDOW_SHORT  = 50
_WINDOW_MEDIUM = 200
_WINDOW_LONG   = 1000


@dataclass
class GameMetricsSnapshot:
    """
    Point-in-time snapshot of game metrics.

    Returned by ``GameMetrics.get_snapshot()`` and served to the API.

    Attributes
    ----------
    step : int
        Episode step at snapshot time.
    game_value : float
        Nash equilibrium value V* (last computed).
    exploitability : float
        Distance from Nash equilibrium (lower = better).
    belief_entropy : float
        Current belief entropy (bits).
    dominant_attacker_type : str
        Attacker type with highest posterior probability.
    dominant_probability : float
        Posterior probability of dominant type.
    attacker_prediction_accuracy : float
        Fraction of steps where predicted attacker action = actual.
    defender_win_rate : float
        Fraction of episodes where defender outscored attacker.
    kill_chain_distribution : dict[str, float]
        Fraction of steps at each kill-chain stage.
    mean_game_value_100 : float
        Mean Nash game value over last 100 steps.
    belief_shift_count : int
        Number of times dominant attacker type changed.
    """
    step:                          int
    game_value:                    float
    exploitability:                float
    belief_entropy:                float
    dominant_attacker_type:        str
    dominant_probability:          float
    attacker_prediction_accuracy:  float
    defender_win_rate:             float
    kill_chain_distribution:       Dict[str, float]
    mean_game_value_100:           float
    belief_shift_count:            int
    n_episodes:                    int          = 0
    n_steps_total:                 int          = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step":                         self.step,
            "game_value":                   round(self.game_value, 4),
            "exploitability":               round(self.exploitability, 4),
            "belief_entropy":               round(self.belief_entropy, 4),
            "dominant_attacker_type":       self.dominant_attacker_type,
            "dominant_probability":         round(self.dominant_probability, 4),
            "attacker_prediction_accuracy": round(self.attacker_prediction_accuracy, 4),
            "defender_win_rate":            round(self.defender_win_rate, 4),
            "kill_chain_distribution":      {
                k: round(v, 4) for k, v in self.kill_chain_distribution.items()
            },
            "mean_game_value_100":          round(self.mean_game_value_100, 4),
            "belief_shift_count":           self.belief_shift_count,
            "n_episodes":                   self.n_episodes,
            "n_steps_total":                self.n_steps_total,
        }


class GameMetrics:
    """
    Tracks and computes game-theoretic metrics across an episode.

    Parameters
    ----------
    max_history : int
        Maximum per-step records to retain in memory.  Default 2000.
    """

    def __init__(self, max_history: int = 2000) -> None:
        self.max_history = max_history

        # Per-step records
        self._step_records:     List[Dict[str, Any]] = []
        self._belief_history:   List[Dict[str, Any]] = []
        self._game_values:      List[float]          = []
        self._exploitabilities: List[float]          = []
        self._entropies:        List[float]          = []

        # Prediction accuracy tracking
        self._prediction_correct: int = 0
        self._prediction_total:   int = 0

        # Episode outcome tracking
        self._episode_outcomes:  List[bool]   = []  # True = defender win
        self._current_step:      int          = 0
        self._n_episodes:        int          = 0

        # Kill-chain stage counts
        self._kc_counts: Dict[str, int] = {s.name: 0 for s in KillChainStage}

        # Belief shift tracking
        self._belief_shift_count: int = 0
        self._last_dominant:      Optional[str] = None

    # ------------------------------------------------------------------ #
    # Recording
    # ------------------------------------------------------------------ #

    def record_step(
        self,
        step:                 int,
        belief:               BeliefState,
        game_state:           GameState,
        nash_eq:              Optional[Any] = None,    # NashEquilibrium
        actual_red_action:    Optional[int] = None,
        predicted_red_action: Optional[int] = None,
    ) -> None:
        """
        Record metrics for a single game step.

        Parameters
        ----------
        step : int
        belief : BeliefState
        game_state : GameState
        nash_eq : NashEquilibrium, optional
        actual_red_action : int, optional
            The red action actually taken.
        predicted_red_action : int, optional
            The red action predicted by the model (argmax of belief strategy).
        """
        self._current_step = step

        # ── Game value and exploitability ──────────────────────────────
        game_value    = 0.0
        exploitability = 0.0
        if nash_eq is not None:
            game_value     = nash_eq.game_value
            exploitability = nash_eq.exploitability
        self._game_values.append(game_value)
        self._exploitabilities.append(exploitability)

        # ── Belief entropy ─────────────────────────────────────────────
        self._entropies.append(belief.entropy)

        # ── Belief shift detection ─────────────────────────────────────
        current_dominant = belief.dominant_type
        if (
            self._last_dominant is not None
            and self._last_dominant != current_dominant
        ):
            self._belief_shift_count += 1
        self._last_dominant = current_dominant

        # ── Kill-chain stage ───────────────────────────────────────────
        stage_name = game_state.kill_chain_stage.name
        self._kc_counts[stage_name] = self._kc_counts.get(stage_name, 0) + 1

        # ── Prediction accuracy ────────────────────────────────────────
        if actual_red_action is not None and predicted_red_action is not None:
            self._prediction_total  += 1
            if actual_red_action == predicted_red_action:
                self._prediction_correct += 1

        # ── Belief history ─────────────────────────────────────────────
        self._belief_history.append({
            "step":          step,
            "probabilities": dict(belief.probabilities),
            "entropy":       belief.entropy,
            "dominant":      current_dominant,
            "game_value":    game_value,
        })

        # Trim to max history
        if len(self._belief_history) > self.max_history:
            self._belief_history.pop(0)

        # Per-step record
        record = {
            "step":                step,
            "game_value":          game_value,
            "exploitability":      exploitability,
            "entropy":             belief.entropy,
            "dominant_type":       current_dominant,
            "dominant_prob":       belief.dominant_probability,
            "n_compromised":       game_state.n_compromised,
            "kill_chain_stage":    stage_name,
            "blue_score":          game_state.blue_score,
            "red_score":           game_state.red_score,
        }
        self._step_records.append(record)
        if len(self._step_records) > self.max_history:
            self._step_records.pop(0)

    def record_episode_end(self, game_state: GameState) -> None:
        """
        Record the outcome of a completed episode.

        Parameters
        ----------
        game_state : GameState
            Final game state.
        """
        self._n_episodes += 1
        defender_win = game_state.blue_score > game_state.red_score
        self._episode_outcomes.append(defender_win)

        # Keep last 1000 episode outcomes
        if len(self._episode_outcomes) > 1000:
            self._episode_outcomes.pop(0)

        logger.debug(
            "Episode %d ended — defender_win=%s, blue=%.2f, red=%.2f",
            self._n_episodes, defender_win,
            game_state.blue_score, game_state.red_score,
        )

    def reset_episode(self) -> None:
        """Reset per-episode tracking (call at episode start)."""
        self._current_step = 0
        self._last_dominant = None

    # ------------------------------------------------------------------ #
    # Snapshot / accessors
    # ------------------------------------------------------------------ #

    def get_snapshot(self) -> GameMetricsSnapshot:
        """
        Return a current-state snapshot of all game metrics.

        Returns
        -------
        GameMetricsSnapshot
        """
        # Game value stats
        last_gv  = self._game_values[-1]   if self._game_values   else 0.0
        last_exp = self._exploitabilities[-1] if self._exploitabilities else 0.0
        mean_gv_100 = float(np.mean(self._game_values[-100:])) if self._game_values else 0.0

        # Belief
        last_entropy = self._entropies[-1] if self._entropies else 0.0
        last_record  = self._step_records[-1] if self._step_records else {}
        dominant     = last_record.get("dominant_type", AttackerType.RANDOM.value)
        dominant_p   = last_record.get("dominant_prob", 1.0 / 3)

        # Prediction accuracy
        pred_acc = (
            self._prediction_correct / self._prediction_total
            if self._prediction_total > 0 else 0.0
        )

        # Defender win rate
        win_rate = (
            sum(self._episode_outcomes[-100:]) / len(self._episode_outcomes[-100:])
            if self._episode_outcomes else 0.0
        )

        # Kill-chain distribution
        total_kc = max(sum(self._kc_counts.values()), 1)
        kc_dist  = {k: v / total_kc for k, v in self._kc_counts.items()}

        return GameMetricsSnapshot(
            step                         = self._current_step,
            game_value                   = last_gv,
            exploitability               = last_exp,
            belief_entropy               = last_entropy,
            dominant_attacker_type       = dominant,
            dominant_probability         = dominant_p,
            attacker_prediction_accuracy = pred_acc,
            defender_win_rate            = win_rate,
            kill_chain_distribution      = kc_dist,
            mean_game_value_100          = mean_gv_100,
            belief_shift_count           = self._belief_shift_count,
            n_episodes                   = self._n_episodes,
            n_steps_total                = len(self._step_records),
        )

    def get_belief_history(
        self, last_n: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Return the belief history for the chart on the dashboard.

        Parameters
        ----------
        last_n : int, optional
            Return only the last N steps.

        Returns
        -------
        list[dict]
            Each entry: ``{step, probabilities, entropy, dominant, game_value}``.
        """
        history = self._belief_history
        if last_n is not None:
            history = history[-last_n:]
        return list(history)

    def get_game_value_curve(
        self, last_n: Optional[int] = None
    ) -> List[float]:
        """Return the game value V* history (for the chart)."""
        gv = self._game_values
        return list(gv[-last_n:] if last_n else gv)

    def get_kill_chain_timeline(self) -> List[Dict[str, Any]]:
        """
        Return the kill-chain stage progression as a timeline.

        Each entry: ``{step, stage_name, stage_idx}``.
        """
        return [
            {
                "step":       r["step"],
                "stage_name": r["kill_chain_stage"],
                "stage_idx":  KillChainStage[r["kill_chain_stage"]].value,
            }
            for r in self._step_records
        ]

    def get_action_prediction_breakdown(self) -> Dict[str, Any]:
        """
        Return per-action prediction accuracy breakdown.

        Returns
        -------
        dict
            Keys: ``n_correct``, ``n_total``, ``accuracy``.
        """
        return {
            "n_correct":  self._prediction_correct,
            "n_total":    self._prediction_total,
            "accuracy":   round(
                self._prediction_correct / max(self._prediction_total, 1), 4
            ),
        }

    def get_attacker_type_timeline(self) -> List[Dict[str, Any]]:
        """
        Return the dominant attacker type at each recorded step.

        Useful for the belief history chart on the dashboard.
        """
        return [
            {
                "step":       r["step"],
                "dominant":   r["dominant_type"],
                "probability":round(r["dominant_prob"], 4),
                "entropy":    round(r["entropy"], 4),
            }
            for r in self._step_records
        ]

    # ------------------------------------------------------------------ #
    # Paper result helpers
    # ------------------------------------------------------------------ #

    def get_paper_metrics(self) -> Dict[str, Any]:
        """
        Return the metrics needed for the paper's game model evaluation table.

        Returns
        -------
        dict
        """
        snapshot = self.get_snapshot()

        return {
            "game_value_mean":             round(float(np.mean(self._game_values or [0])), 4),
            "game_value_std":              round(float(np.std(self._game_values or [0])), 4),
            "exploitability_mean":         round(float(np.mean(self._exploitabilities or [0])), 4),
            "belief_entropy_mean":         round(float(np.mean(self._entropies or [0])), 4),
            "attacker_prediction_accuracy":round(snapshot.attacker_prediction_accuracy, 4),
            "belief_shift_count":          self._belief_shift_count,
            "defender_win_rate":           round(snapshot.defender_win_rate, 4),
            "n_episodes":                  self._n_episodes,
            "n_steps_total":               len(self._step_records),
        }

    def reset_all(self) -> None:
        """Reset ALL metrics (use between experiments, not between episodes)."""
        self._step_records.clear()
        self._belief_history.clear()
        self._game_values.clear()
        self._exploitabilities.clear()
        self._entropies.clear()
        self._prediction_correct  = 0
        self._prediction_total    = 0
        self._episode_outcomes.clear()
        self._kc_counts = {s.name: 0 for s in KillChainStage}
        self._belief_shift_count  = 0
        self._last_dominant       = None
        self._current_step        = 0
        self._n_episodes          = 0

    def __repr__(self) -> str:
        return (
            f"GameMetrics("
            f"steps={len(self._step_records)}, "
            f"episodes={self._n_episodes}, "
            f"belief_shifts={self._belief_shift_count})"
        )