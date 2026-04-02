"""
envs/cyborg_wrapper.py
======================
Primary Gym-compatible wrapper around the CybORG cyber operations simulator.

This is the single object that all training scripts, the API, and SB3
interact with.  It composes the other env modules into a unified interface:

    CybORGWrapper
        ├── ScenarioLoader     → loads the raw CybORG env
        ├── ObservationProcessor → raw obs dict → 54-dim float32 vector
        ├── ActionMapper       → integer → CybORG action object
        ├── RewardShaper       → raw reward → shaped reward
        └── DriftInjector      → injects concept drift at scheduled steps

Gym interface
-------------
    env = CybORGWrapper(config)
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(action)

The environment follows the Gymnasium (not gym) API (terminated + truncated).

Usage
-----
    from envs import make_env

    env = make_env(config)

    # Or directly:
    env = CybORGWrapper(config={
        "scenario":        "scenario2",
        "red_agent":       "B_lineAgent",
        "max_steps":       100,
        "reward_shaping":  True,
        "drift": {
            "mode":        "attacker_switch",
            "drift_steps": [250_000, 500_000],
        },
    })
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym

from envs.observation_space import ObservationProcessor
from envs.action_space import ActionMapper
from envs.reward_shaper import RewardShaper
from envs.scenario_loader import ScenarioLoader
from envs.drift_injector import DriftInjector

logger = logging.getLogger(__name__)


class CybORGWrapper(gym.Env):
    """
    Gym-compatible wrapper around the CybORG cyber simulation environment.

    Converts CybORG's internal API into a clean Gymnasium interface,
    handling observation flattening, action mapping, reward shaping,
    and drift injection transparently.

    Parameters
    ----------
    config : dict
        Environment configuration.  Keys:

        scenario : str
            Scenario name: ``"scenario1"`` or ``"scenario2"`` (default).
        red_agent : str
            Red agent type.  Default ``"B_lineAgent"``.
        max_steps : int
            Episode horizon.  Default 100.
        reward_shaping : bool
            If True, wrap raw rewards with ``RewardShaper``.  Default True.
        reward_shaper_config : dict, optional
            Passed to ``RewardShaper.__init__``.
        drift : dict, optional
            Passed to ``DriftInjector.__init__``.  If absent, no drift.
        seed : int, optional
            Random seed for reproducibility.

    Attributes
    ----------
    observation_space : gym.spaces.Box
        Shape (54,), dtype float32, values in [0, 1].
    action_space : gym.spaces.Discrete
        54 discrete actions for Scenario2.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        self.config = config or {}

        # ── Load scenario via ScenarioLoader ──────────────────────────
        scenario  = self.config.get("scenario", "scenario2")
        red_agent = self.config.get("red_agent", "B_lineAgent")
        max_steps = self.config.get("max_steps", 100)
        seed      = self.config.get("seed")

        self._scenario_name = scenario
        self._red_agent     = red_agent
        self.max_steps      = max_steps

        self._loader = ScenarioLoader(
            scenario_dir=self.config.get("scenario_dir", "data/scenarios")
        )
        self._cyborg = self._loader.load(
            scenario  = scenario,
            red_agent = red_agent,
            max_steps = max_steps,
            seed      = seed,
        )

        meta = self._loader.get_metadata(scenario)

        # ── Sub-modules ────────────────────────────────────────────────
        self._obs_processor = ObservationProcessor(
            n_hosts   = meta["n_hosts"],
            max_steps = max_steps,
        )

        self._action_mapper = ActionMapper(scenario=scenario)

        self._reward_shaper: Optional[RewardShaper] = None
        if self.config.get("reward_shaping", True):
            self._reward_shaper = RewardShaper(
                config=self.config.get("reward_shaper_config")
            )

        drift_cfg = self.config.get("drift")
        self._drift_injector: Optional[DriftInjector] = (
            DriftInjector(config=drift_cfg) if drift_cfg else None
        )

        # ── Gym spaces ─────────────────────────────────────────────────
        self.observation_space = self._obs_processor.observation_space
        self.action_space      = self._action_mapper.action_space

        # ── Episode state ──────────────────────────────────────────────
        self._step:             int = 0
        self._global_step:      int = 0
        self._episode_count:    int = 0
        self._last_obs:         Optional[np.ndarray] = None
        self._last_action_success: bool = True
        self._last_action_type:    str = "Monitor"
        self._attacker_host_idx:   int = 0
        self._episode_rewards:     List[float] = []
        self._current_ep_reward:   float = 0.0

        logger.info(
            "CybORGWrapper ready — scenario=%r, obs_dim=%d, n_actions=%d",
            scenario,
            self._obs_processor.obs_dim,
            self._action_mapper.n_actions,
        )

    # ------------------------------------------------------------------ #
    # Gym interface
    # ------------------------------------------------------------------ #

    def reset(
        self,
        seed:    Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to the start of a new episode.

        Parameters
        ----------
        seed : int, optional
            RNG seed for this episode.
        options : dict, optional
            Unused; included for Gymnasium API compatibility.

        Returns
        -------
        obs : np.ndarray
            Initial observation vector, shape (54,).
        info : dict
            Episode metadata: scenario, red_agent, max_steps, episode_count.
        """
        # Track episode stats
        if self._current_ep_reward != 0.0:
            self._episode_rewards.append(self._current_ep_reward)
        self._current_ep_reward = 0.0

        self._step          = 0
        self._episode_count += 1
        self._last_action_success = True
        self._last_action_type    = "Monitor"
        self._attacker_host_idx   = 0

        # Reset sub-modules
        self._obs_processor.reset()
        if self._reward_shaper:
            self._reward_shaper.reset()

        # Reset CybORG — v2.1 API: reset(agent) returns Results object
        if seed is not None:
            try:
                self._cyborg.set_seed(seed)
            except AttributeError:
                pass

        try:
            result = self._cyborg.reset(agent="Blue")
        except TypeError:
            result = self._cyborg.reset()

        # Extract raw observation from Results object or dict
        raw_obs = self._extract_obs_from_result(result)

        obs = self._process_obs(raw_obs, episode_start=True)
        self._last_obs = obs

        info = self._make_info(raw_obs=raw_obs, episode_start=True)

        return obs, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one environment step.

        Parameters
        ----------
        action : int
            Integer action index from the Discrete action space.

        Returns
        -------
        obs : np.ndarray
            Next observation vector, shape (54,).
        reward : float
            Shaped (or raw) reward for this step.
        terminated : bool
            True if the episode ended due to environment logic
            (e.g. max_steps reached or all hosts compromised).
        truncated : bool
            Always False (we use terminated for both cases).
        info : dict
            Step metadata including raw_reward, action_description, success.
        """
        self._step       += 1
        self._global_step += 1

        # ── Map integer action → CybORG action ────────────────────────
        cyborg_action = self._action_mapper.to_cyborg_action(
            action_idx = int(action),
            cyborg_env = self._cyborg,
        )

        action_spec = self._action_mapper.get_spec(int(action))
        self._last_action_type = action_spec.action_type

        # ── Step CybORG ────────────────────────────────────────────────
        try:
            result = self._cyborg.step(action=cyborg_action, agent="Blue")
            raw_obs, raw_reward, done, info = self._parse_cyborg_result(result)
        except Exception as exc:
            logger.error("CybORG step failed: %s", exc)
            raw_obs    = {}
            raw_reward = -1.0
            done       = True
            info       = {"error": str(exc)}

        self._last_action_success = info.get("action_success", True)
        self._attacker_host_idx   = info.get("attacker_host_idx", 0)

        prev_obs = self._last_obs if self._last_obs is not None else np.zeros(
            self._obs_processor.obs_dim, dtype=np.float32
        )

        # ── Process observation ────────────────────────────────────────
        obs = self._process_obs(raw_obs)
        self._last_obs = obs

        # ── Shape reward ───────────────────────────────────────────────
        shaped_reward = raw_reward
        if self._reward_shaper is not None:
            shaped_reward = self._reward_shaper.shape(
                raw_reward      = raw_reward,
                prev_obs        = prev_obs,
                curr_obs        = obs,
                action_idx      = int(action),
                action_mapper   = self._action_mapper,
                step            = self._step,
                max_steps       = self.max_steps,
                action_success  = self._last_action_success,
            )

        # Apply drift-level reward transformation
        if self._drift_injector is not None:
            shaped_reward = self._drift_injector.apply_reward_shaping(shaped_reward)

        self._current_ep_reward += shaped_reward

        # ── Termination ────────────────────────────────────────────────
        terminated = done or self._step >= self.max_steps
        truncated  = False

        step_info = {
            **info,
            "raw_reward":          raw_reward,
            "shaped_reward":       shaped_reward,
            "step":                self._step,
            "global_step":         self._global_step,
            "action_description":  action_spec.description,
            "action_success":      self._last_action_success,
            "episode_reward_so_far": self._current_ep_reward,
            "drift_events":        (
                self._drift_injector.drift_history
                if self._drift_injector else []
            ),
        }

        if self._reward_shaper:
            step_info["reward_breakdown"] = self._reward_shaper.get_last_breakdown()

        return obs, float(shaped_reward), terminated, truncated, step_info

    def close(self) -> None:
        """Release CybORG resources."""
        try:
            self._cyborg.close()
        except AttributeError:
            pass

    # ------------------------------------------------------------------ #
    # Utility methods
    # ------------------------------------------------------------------ #

    def action_masks(self) -> np.ndarray:
        """
        Return a boolean mask of currently valid actions.

        Used by action-masked PPO variants to avoid training on
        provably illegal actions.

        Returns
        -------
        np.ndarray
            Shape (n_actions,), dtype bool.
        """
        if self._last_obs is None:
            return np.ones(self._action_mapper.n_actions, dtype=bool)

        # Decode current obs to get compromised / decoy host lists
        decoded = self._obs_processor.decode(self._last_obs)
        compromised = [
            h for h, s in decoded["hosts"].items() if s["compromised"]
        ]
        decoy_hosts = [
            h for h, s in decoded["hosts"].items() if s["is_decoy"]
        ]

        return self._action_mapper.valid_action_mask(
            compromised_hosts = compromised,
            decoy_hosts       = decoy_hosts,
        )

    def get_network_state(self) -> Dict[str, Any]:
        """
        Return a decoded snapshot of the current network state.

        Used by the ``/network/topology`` API route to feed the dashboard.

        Returns
        -------
        dict
            ``{"hosts": {...}, "action_feedback": {...}}``.
        """
        if self._last_obs is None:
            return {}
        return self._obs_processor.decode(self._last_obs)

    def get_metrics(self) -> Dict[str, Any]:
        """Return current episode and global training metrics."""
        avg_reward = (
            float(np.mean(self._episode_rewards[-100:]))
            if self._episode_rewards else 0.0
        )
        return {
            "scenario":          self._scenario_name,
            "red_agent":         self._red_agent,
            "episode_count":     self._episode_count,
            "global_step":       self._global_step,
            "current_step":      self._step,
            "max_steps":         self.max_steps,
            "current_ep_reward": round(self._current_ep_reward, 4),
            "mean_ep_reward_100": round(avg_reward, 4),
            "drift_events":      (
                self._drift_injector.n_drift_events
                if self._drift_injector else 0
            ),
        }

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _process_obs(
        self,
        raw_obs:       Any,
        episode_start: bool = False,
    ) -> np.ndarray:
        """Convert raw CybORG obs to processed vector, applying drift."""
        obs = self._obs_processor.process(
            raw_obs            = raw_obs if isinstance(raw_obs, dict) else {},
            step               = self._step,
            last_action_success= self._last_action_success,
            last_action_type   = self._last_action_type,
            attacker_host_idx  = self._attacker_host_idx,
        )

        # Apply drift injection (may modify obs vector)
        if self._drift_injector is not None:
            obs = self._drift_injector.maybe_inject(
                obs           = obs,
                global_step   = self._global_step,
                cyborg_env    = self,
                episode_start = episode_start,
            )

        return obs

    @staticmethod
    def _extract_obs_from_result(result: Any) -> Dict[str, Any]:
        """
        Extract the observation dict from a CybORG Results object,
        a raw dict, or fallback gracefully.

        CybORG v2.1's Results has ``.observation`` which may be:
          - A dict (the raw observation)
          - An ``Observation`` object with a ``.data`` attribute
          - None
        """
        if isinstance(result, dict):
            return result

        # CybORG v2.1 Results object
        obs = getattr(result, "observation", None)
        if obs is None:
            return {}

        # Observation object → extract .data dict
        if hasattr(obs, "data"):
            return obs.data if isinstance(obs.data, dict) else {}

        if isinstance(obs, dict):
            return obs

        return {}

    @staticmethod
    def _parse_cyborg_result(result: Any) -> Tuple[Dict, float, bool, Dict]:
        """
        Parse the return value from CybORG.step() into (obs, reward, done, info).

        Handles CybORG v2.1 Results objects (primary), tuples (legacy/mock),
        and graceful fallbacks.
        """
        # ── Tuple return (MockCybORG or legacy wrappers) ───────────────
        if isinstance(result, tuple):
            if len(result) == 4:
                obs, reward, done, info = result
                return obs or {}, float(reward), bool(done), info or {}
            elif len(result) == 5:
                obs, reward, terminated, truncated, info = result
                return obs or {}, float(reward), bool(terminated or truncated), info or {}

        # ── CybORG v2.1 Results object ─────────────────────────────────
        try:
            obs_raw = getattr(result, "observation", None)
            if obs_raw is None:
                obs_dict: Dict = {}
            elif hasattr(obs_raw, "data"):
                obs_dict = obs_raw.data if isinstance(obs_raw.data, dict) else {}
            elif isinstance(obs_raw, dict):
                obs_dict = obs_raw
            else:
                obs_dict = {}

            reward = float(getattr(result, "reward", 0.0) or 0.0)
            done = bool(getattr(result, "done", False))
            info: Dict[str, Any] = {
                "action_success": not getattr(result, "error", None),
                "action_name":    getattr(result, "action_name", None),
            }
            return obs_dict, reward, done, info

        except Exception:
            return {}, 0.0, False, {}

    def _make_info(
        self, raw_obs: Any = None, episode_start: bool = False
    ) -> Dict[str, Any]:
        """Build the info dict returned by ``reset()``."""
        return {
            "scenario":        self._scenario_name,
            "red_agent":       self._red_agent,
            "max_steps":       self.max_steps,
            "episode_count":   self._episode_count,
            "obs_dim":         self._obs_processor.obs_dim,
            "n_actions":       self._action_mapper.n_actions,
            "episode_start":   episode_start,
        }

    def __repr__(self) -> str:
        return (
            f"CybORGWrapper("
            f"scenario={self._scenario_name!r}, "
            f"obs_dim={self._obs_processor.obs_dim}, "
            f"n_actions={self._action_mapper.n_actions}, "
            f"episode={self._episode_count})"
        )