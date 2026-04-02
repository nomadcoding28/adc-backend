"""
envs/env_factory.py
===================
Top-level factory for building ACD environments from config dicts.

This is the single entry point for all environment construction.
Training scripts, the API, and the evaluation pipeline all call
``make_env()`` or ``EnvFactory.build()`` — they never import
``CybORGWrapper`` directly.

Usage
-----
    from envs import make_env

    # Simple (single env, no drift)
    env = make_env(config)

    # Vectorised (4 parallel workers)
    env = make_env(config, n_envs=4)

    # Eval env (no reward shaping, no drift, deterministic)
    env = make_env(config, mode="eval")

    # Using the class directly
    env = EnvFactory.build(config, mode="train", n_envs=8)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Literal, Optional, Union

from stable_baselines3.common.vec_env import VecEnv

from envs.cyborg_wrapper import CybORGWrapper
from envs.multi_env import make_vec_env, SingleEnvWrapper

logger = logging.getLogger(__name__)

# Build mode type alias
EnvMode = Literal["train", "eval", "debug"]


def make_env(
    config: Dict[str, Any],
    n_envs: int = 1,
    mode:   EnvMode = "train",
    seed:   Optional[int] = None,
) -> Union[CybORGWrapper, VecEnv]:
    """
    Build a CybORG environment from a config dict.

    Parameters
    ----------
    config : dict
        Full config dict.  Environment config is read from
        ``config["env"]`` if present, otherwise the whole dict is used.
    n_envs : int
        Number of parallel workers.  1 = single env (no multiprocessing).
    mode : str
        ``"train"``  — reward shaping ON, drift injection ON (if configured).
        ``"eval"``   — reward shaping OFF, drift injection OFF.
        ``"debug"``  — single env, verbose logging, drift OFF.
    seed : int, optional
        Base seed.  Each worker gets seed + worker_idx.

    Returns
    -------
    CybORGWrapper or VecEnv
        If n_envs=1, returns a ``CybORGWrapper``.
        If n_envs>1, returns a ``SubprocVecEnv`` wrapping N workers.
    """
    env_cfg = _extract_env_config(config, mode=mode, seed=seed)

    if n_envs == 1:
        logger.info("Building single CybORGWrapper (mode=%r)", mode)
        return CybORGWrapper(env_cfg)

    logger.info(
        "Building vectorised env — n_envs=%d, mode=%r", n_envs, mode
    )
    return make_vec_env(
        config  = env_cfg,
        n_envs  = n_envs,
        seed    = seed,
        monitor = mode == "train",
    )


class EnvFactory:
    """
    Stateful factory that caches scenario metadata and reuses it across
    multiple ``build()`` calls.

    Useful when the API creates multiple evaluation environments in the
    same process — avoids re-loading scenario YAML files each time.

    Parameters
    ----------
    config : dict
        Full application config dict.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self._cached_metadata: Dict[str, Any] = {}

    def build(
        self,
        n_envs: int = 1,
        mode:   EnvMode = "train",
        seed:   Optional[int] = None,
    ) -> Union[CybORGWrapper, VecEnv]:
        """
        Build an environment using the factory's stored config.

        Parameters
        ----------
        n_envs : int
            Number of parallel workers.
        mode : str
            ``"train"``, ``"eval"``, or ``"debug"``.
        seed : int, optional
            Base seed for reproducibility.

        Returns
        -------
        CybORGWrapper or VecEnv
        """
        return make_env(self.config, n_envs=n_envs, mode=mode, seed=seed)

    def build_eval_env(self, seed: Optional[int] = 42) -> CybORGWrapper:
        """
        Convenience: build a single deterministic evaluation environment.

        Reward shaping and drift are both disabled.  Used by the evaluation
        pipeline and benchmark runner.
        """
        return self.build(n_envs=1, mode="eval", seed=seed)

    def build_train_env(
        self, n_envs: int = 4, seed: Optional[int] = None
    ) -> VecEnv:
        """
        Convenience: build a vectorised training environment.

        Reward shaping and drift are enabled based on the config.
        """
        return self.build(n_envs=n_envs, mode="train", seed=seed)

    def get_scenario_info(self) -> Dict[str, Any]:
        """
        Return scenario metadata (hosts, n_actions, obs_dim, etc.)
        without instantiating a full environment.
        """
        from envs.scenario_loader import ScenarioLoader
        scenario = self.config.get("env", self.config).get("scenario", "scenario2")
        if scenario not in self._cached_metadata:
            loader = ScenarioLoader()
            self._cached_metadata[scenario] = loader.get_metadata(scenario)
        return self._cached_metadata[scenario]


# ── Private helpers ─────────────────────────────────────────────────────────

def _extract_env_config(
    config:  Dict[str, Any],
    mode:    EnvMode,
    seed:    Optional[int],
) -> Dict[str, Any]:
    """
    Extract and modify the environment config section based on build mode.

    - If ``config`` has an ``"env"`` key, uses that sub-dict.
    - Mode overrides: eval turns off shaping + drift; debug sets verbose logging.
    """
    # Use env sub-section if available, else treat whole config as env config
    env_cfg = dict(config.get("env", config))

    if seed is not None:
        env_cfg["seed"] = seed

    if mode == "eval":
        # Clean eval: no reward shaping, no drift, no random seed interference
        env_cfg["reward_shaping"] = False
        env_cfg.pop("drift", None)
        if "seed" not in env_cfg:
            env_cfg["seed"] = 42

    elif mode == "debug":
        # Debug: single env, no drift, verbose logging
        env_cfg.pop("drift", None)
        env_cfg.setdefault("reward_shaping", False)
        logging.getLogger("envs").setLevel(logging.DEBUG)

    # mode == "train": use config as-is (drift and shaping from yaml)

    return env_cfg