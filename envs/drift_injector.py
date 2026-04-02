"""
envs/drift_injector.py
======================
Synthetically injects concept drift into training episodes.

In real networks, attack patterns change over time — new malware families,
different attacker tools, different target selection strategies.  The drift
injector simulates these distributional shifts during training so the
continual learning pipeline has meaningful drift events to respond to.

Drift modes
-----------
  attacker_switch   : Switch the red agent type mid-training
                      (e.g. B_lineAgent → RedMeanderAgent)
  observation_shift : Apply a linear bias to the observation vector,
                      simulating sensor calibration drift
  reward_shift      : Scale or shift reward values to simulate
                      environment change with same obs space
  composite         : Apply multiple drift types simultaneously

Usage
-----
    injector = DriftInjector(config={
        "mode":              "attacker_switch",
        "drift_step":        250_000,       # trigger at step 250k
        "drift_probability": 0.0,           # deterministic if 0
        "new_red_agent":     "RedMeanderAgent",
    })

    # Called from the environment wrapper every step
    drifted_obs = injector.maybe_inject(
        obs          = current_obs,
        global_step  = timestep_counter,
        cyborg_env   = cyborg_env_object,
    )

    # Check if drift has been applied
    if injector.has_drifted:
        print(f"Drift triggered at step {injector.drift_triggered_at}")
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Drift mode constants ────────────────────────────────────────────────────
MODE_ATTACKER_SWITCH    = "attacker_switch"
MODE_OBSERVATION_SHIFT  = "observation_shift"
MODE_REWARD_SHIFT       = "reward_shift"
MODE_COMPOSITE          = "composite"
MODE_NONE               = "none"

_SUPPORTED_MODES = {
    MODE_ATTACKER_SWITCH,
    MODE_OBSERVATION_SHIFT,
    MODE_REWARD_SHIFT,
    MODE_COMPOSITE,
    MODE_NONE,
}


class DriftEvent:
    """
    Immutable record of a single drift event.

    Attributes
    ----------
    step : int
        Global timestep at which drift was triggered.
    mode : str
        Drift mode that was applied.
    description : str
        Human-readable description of the change.
    metadata : dict
        Additional details (e.g. old/new red agent, shift magnitude).
    """

    __slots__ = ("step", "mode", "description", "metadata")

    def __init__(
        self,
        step:        int,
        mode:        str,
        description: str,
        metadata:    Optional[Dict[str, Any]] = None,
    ) -> None:
        self.step        = step
        self.mode        = mode
        self.description = description
        self.metadata    = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step":        self.step,
            "mode":        self.mode,
            "description": self.description,
            "metadata":    self.metadata,
        }

    def __repr__(self) -> str:
        return (
            f"DriftEvent(step={self.step}, mode={self.mode!r}, "
            f"description={self.description!r})"
        )


class DriftInjector:
    """
    Controls when and how concept drift is injected into training.

    Supports both *scheduled* drift (at a fixed step) and *probabilistic*
    drift (triggered with a given probability per episode).

    Multiple drift events can be configured by passing a list of step
    thresholds.  After each drift event, the internal state resets
    to the next threshold.

    Parameters
    ----------
    config : dict
        Drift configuration.  Keys:

        mode : str
            One of ``"attacker_switch"``, ``"observation_shift"``,
            ``"reward_shift"``, ``"composite"``, ``"none"``.

        drift_steps : list[int] or int
            Global step(s) at which to trigger drift.
            If a single int, only one drift event is scheduled.

        drift_probability : float
            If > 0, drift is triggered with this probability at the start
            of each episode (ignores ``drift_steps``).

        obs_shift_magnitude : float
            Magnitude of observation bias for ``observation_shift`` mode.
            Default 0.1.

        obs_shift_dims : list[int], optional
            Which observation dimensions to shift.  Defaults to all.

        reward_scale : float
            Reward scaling factor for ``reward_shift`` mode.  Default 0.8.

        new_red_agent : str
            Red agent class name to switch to for ``attacker_switch`` mode.
            Default ``"RedMeanderAgent"``.

        on_drift : callable, optional
            Callback ``(DriftEvent) -> None`` fired when drift is triggered.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}

        self.mode = cfg.get("mode", MODE_NONE)
        if self.mode not in _SUPPORTED_MODES:
            raise ValueError(
                f"Unknown drift mode {self.mode!r}. "
                f"Supported: {_SUPPORTED_MODES}"
            )

        # Parse drift_steps into a sorted list
        raw_steps = cfg.get("drift_steps", cfg.get("drift_step", []))
        if isinstance(raw_steps, int):
            raw_steps = [raw_steps]
        self._scheduled_steps: List[int] = sorted(set(raw_steps))

        self.drift_probability  = float(cfg.get("drift_probability", 0.0))
        self.obs_shift_mag      = float(cfg.get("obs_shift_magnitude", 0.1))
        self.obs_shift_dims: Optional[List[int]] = cfg.get("obs_shift_dims")
        self.reward_scale       = float(cfg.get("reward_scale", 0.8))
        self.new_red_agent      = cfg.get("new_red_agent", "RedMeanderAgent")
        self._on_drift_cb: Optional[Callable] = cfg.get("on_drift")

        # Mutable state
        self._drift_events:     List[DriftEvent] = []
        self._next_step_idx:    int = 0           # pointer into _scheduled_steps
        self._obs_bias:         Optional[np.ndarray] = None
        self._reward_scale_cur: float = 1.0
        self._rng               = np.random.default_rng()

        logger.debug(
            "DriftInjector: mode=%r, scheduled_steps=%s, prob=%.3f",
            self.mode, self._scheduled_steps, self.drift_probability,
        )

    # ------------------------------------------------------------------ #
    # Primary interface
    # ------------------------------------------------------------------ #

    @property
    def has_drifted(self) -> bool:
        """True if at least one drift event has been triggered."""
        return len(self._drift_events) > 0

    @property
    def n_drift_events(self) -> int:
        """Total number of drift events triggered so far."""
        return len(self._drift_events)

    @property
    def drift_history(self) -> List[Dict[str, Any]]:
        """Full list of DriftEvent dicts (for API / dashboard)."""
        return [e.to_dict() for e in self._drift_events]

    @property
    def drift_triggered_at(self) -> Optional[int]:
        """Global step of the most recent drift event, or None."""
        return self._drift_events[-1].step if self._drift_events else None

    def maybe_inject(
        self,
        obs:         np.ndarray,
        global_step: int,
        cyborg_env:  Optional[Any] = None,
        episode_start: bool = False,
    ) -> np.ndarray:
        """
        Optionally apply drift to the current observation.

        Called every step from the environment wrapper.  Returns either
        the original observation (no drift) or a modified one.

        Parameters
        ----------
        obs : np.ndarray
            Current processed observation vector, shape (54,).
        global_step : int
            Total environment steps elapsed across all episodes.
        cyborg_env : CybORG env, optional
            Needed for ``attacker_switch`` mode to reload the environment.
        episode_start : bool
            If True, also check probabilistic drift trigger.

        Returns
        -------
        np.ndarray
            Observation (possibly modified by drift).
        """
        if self.mode == MODE_NONE:
            return obs

        triggered = False

        # ── Scheduled drift check ──────────────────────────────────────
        if (
            self._next_step_idx < len(self._scheduled_steps)
            and global_step >= self._scheduled_steps[self._next_step_idx]
        ):
            triggered = True
            self._next_step_idx += 1

        # ── Probabilistic drift check (episode-level) ──────────────────
        elif (
            episode_start
            and self.drift_probability > 0
            and self._rng.random() < self.drift_probability
        ):
            triggered = True

        if triggered:
            self._trigger_drift(global_step, cyborg_env)

        # ── Apply persistent observation shift (if active) ─────────────
        if self._obs_bias is not None:
            obs = np.clip(obs + self._obs_bias, 0.0, 1.0)

        return obs

    def apply_reward_shaping(self, reward: float) -> float:
        """
        Apply any active reward-level drift transformation.

        Parameters
        ----------
        reward : float
            Raw (possibly already shaped) reward.

        Returns
        -------
        float
            Transformed reward.
        """
        return reward * self._reward_scale_cur

    def reset(self, seed: Optional[int] = None) -> None:
        """
        Reset injector state for a new training run.

        .. note::
            This resets the drift schedule pointer too, so drift events
            will be re-triggered from the start.  Do NOT call this between
            episodes within a training run.
        """
        self._drift_events.clear()
        self._next_step_idx   = 0
        self._obs_bias        = None
        self._reward_scale_cur = 1.0
        if seed is not None:
            self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ #
    # Internal drift application
    # ------------------------------------------------------------------ #

    def _trigger_drift(
        self,
        step:       int,
        cyborg_env: Optional[Any] = None,
    ) -> None:
        """Execute the configured drift mode at the current step."""
        logger.info(
            "Concept drift triggered at step %d (mode=%r, event #%d)",
            step, self.mode, len(self._drift_events) + 1,
        )

        if self.mode == MODE_ATTACKER_SWITCH:
            event = self._apply_attacker_switch(step, cyborg_env)

        elif self.mode == MODE_OBSERVATION_SHIFT:
            event = self._apply_obs_shift(step)

        elif self.mode == MODE_REWARD_SHIFT:
            event = self._apply_reward_shift(step)

        elif self.mode == MODE_COMPOSITE:
            event = self._apply_composite(step, cyborg_env)

        else:
            return

        self._drift_events.append(event)

        if self._on_drift_cb is not None:
            try:
                self._on_drift_cb(event)
            except Exception as exc:
                logger.warning("on_drift callback raised: %s", exc)

    def _apply_attacker_switch(
        self, step: int, cyborg_env: Optional[Any]
    ) -> DriftEvent:
        """
        Switch the red agent type to simulate an attack pattern change.

        In a real training run, this requires resetting the CybORG env
        with the new agent.  The CybORGWrapper handles the actual reset;
        this method just records the event and sets a flag.
        """
        prev_agent = "B_lineAgent"   # assumed initial

        if cyborg_env is not None:
            try:
                # CybORG 2.x: reload with new red agent
                from envs.scenario_loader import ScenarioLoader
                loader = ScenarioLoader()
                new_env = loader.load(
                    red_agent="RedMeanderAgent"
                    if self.new_red_agent == "RedMeanderAgent"
                    else self.new_red_agent
                )
                # Transfer the step count back to the wrapper
                cyborg_env._cyborg = new_env
                logger.info("Switched red agent to %r", self.new_red_agent)
            except Exception as exc:
                logger.warning("Attacker switch failed: %s", exc)

        return DriftEvent(
            step        = step,
            mode        = MODE_ATTACKER_SWITCH,
            description = f"Red agent switched: {prev_agent} → {self.new_red_agent}",
            metadata    = {
                "prev_agent": prev_agent,
                "new_agent":  self.new_red_agent,
            },
        )

    def _apply_obs_shift(self, step: int) -> DriftEvent:
        """
        Add a persistent bias vector to all future observations.

        Models sensor recalibration drift — the raw CybORG values are
        identical but the Blue agent's view shifts by a constant bias.
        """
        from envs.observation_space import OBS_DIM

        dims = self.obs_shift_dims or list(range(OBS_DIM))
        bias = np.zeros(OBS_DIM, dtype=np.float32)

        # Alternate bias direction on successive drift events to simulate
        # back-and-forth oscillation
        direction = (-1) ** len(self._drift_events)
        bias[dims] = direction * self.obs_shift_mag

        self._obs_bias = bias

        return DriftEvent(
            step        = step,
            mode        = MODE_OBSERVATION_SHIFT,
            description = (
                f"Observation shift applied: magnitude={self.obs_shift_mag:.3f}, "
                f"dims={'all' if not self.obs_shift_dims else self.obs_shift_dims}"
            ),
            metadata = {
                "magnitude":  self.obs_shift_mag,
                "n_dims":     len(dims),
                "direction":  direction,
            },
        )

    def _apply_reward_shift(self, step: int) -> DriftEvent:
        """
        Scale the reward signal to simulate a changed environment dynamic.

        E.g. environment now penalises breaches more heavily — same obs,
        different reward scale.
        """
        old_scale = self._reward_scale_cur
        self._reward_scale_cur = self.reward_scale

        return DriftEvent(
            step        = step,
            mode        = MODE_REWARD_SHIFT,
            description = (
                f"Reward scale changed: {old_scale:.2f} → {self.reward_scale:.2f}"
            ),
            metadata = {
                "old_scale": old_scale,
                "new_scale": self.reward_scale,
            },
        )

    def _apply_composite(
        self, step: int, cyborg_env: Optional[Any]
    ) -> DriftEvent:
        """Apply both observation shift and attacker switch simultaneously."""
        self._apply_obs_shift(step)
        attacker_event = self._apply_attacker_switch(step, cyborg_env)

        return DriftEvent(
            step        = step,
            mode        = MODE_COMPOSITE,
            description = (
                f"Composite drift: obs_shift + {attacker_event.description}"
            ),
            metadata = {
                "obs_shift_magnitude": self.obs_shift_mag,
                **attacker_event.metadata,
            },
        )

    # ------------------------------------------------------------------ #
    # Utilities
    # ------------------------------------------------------------------ #

    def get_config(self) -> Dict[str, Any]:
        """Return current configuration as a serialisable dict."""
        return {
            "mode":               self.mode,
            "scheduled_steps":    self._scheduled_steps,
            "drift_probability":  self.drift_probability,
            "obs_shift_mag":      self.obs_shift_mag,
            "reward_scale":       self.reward_scale,
            "new_red_agent":      self.new_red_agent,
            "n_events_triggered": self.n_drift_events,
        }

    def __repr__(self) -> str:
        return (
            f"DriftInjector("
            f"mode={self.mode!r}, "
            f"scheduled_steps={self._scheduled_steps}, "
            f"events_triggered={self.n_drift_events})"
        )