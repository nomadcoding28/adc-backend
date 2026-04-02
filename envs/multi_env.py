"""
envs/multi_env.py
=================
Vectorised environment wrapper for parallel rollout collection.

Training is significantly faster with multiple environment instances
running in parallel — SB3's SubprocVecEnv spawns N worker processes
each running a CybORGWrapper, and the main process collects rollouts
from all of them simultaneously.

Primary function
----------------
    make_vec_env(config, n_envs) → VecEnv

Usage
-----
    from envs import make_vec_env

    vec_env = make_vec_env(config, n_envs=4)

    # Use as a regular SB3 VecEnv
    agent = ACDPPOAgent(vec_env, agent_config)
    agent.learn(total_timesteps=1_000_000)

Performance notes
-----------------
- Use ``n_envs=1`` during debugging (avoids subprocess pickling issues).
- Use ``n_envs=4`` or ``8`` for training on a multi-core CPU.
- On GPU, observation processing is fast enough that ``n_envs=4`` is
  typically optimal — more workers add IPC overhead.
- CybORG is not thread-safe, so always use ``SubprocVecEnv`` (not
  ``DummyVecEnv``) for n_envs > 1.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, List, Optional

from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecEnv,
    VecMonitor,
    VecNormalize,
)

from envs.cyborg_wrapper import CybORGWrapper

logger = logging.getLogger(__name__)


def make_vec_env(
    config:       Dict[str, Any],
    n_envs:       int = 1,
    seed:         Optional[int] = None,
    normalize_obs: bool = False,
    monitor:      bool = True,
    vec_env_cls:  Optional[type] = None,
) -> VecEnv:
    """
    Create a vectorised CybORGWrapper environment for parallel training.

    Parameters
    ----------
    config : dict
        Environment configuration dict (passed to each CybORGWrapper).
        Each worker gets its own copy with a unique seed offset.
    n_envs : int
        Number of parallel environment workers.  Default 1.
        For n_envs=1, uses DummyVecEnv (simpler, easier to debug).
        For n_envs>1, uses SubprocVecEnv.
    seed : int, optional
        Base random seed.  Worker i uses seed + i for reproducibility.
    normalize_obs : bool
        If True, wrap with VecNormalize to normalise observations and
        rewards using running mean/std.  Usually not needed since our
        ObservationProcessor already normalises to [0, 1].
    monitor : bool
        If True, wrap with VecMonitor to track episode rewards/lengths.
        These stats are logged to TensorBoard during training.
    vec_env_cls : type, optional
        Override the VecEnv class.  Defaults to DummyVecEnv (n_envs=1)
        or SubprocVecEnv (n_envs>1).

    Returns
    -------
    VecEnv
        Ready-to-use vectorised environment.

    Raises
    ------
    ValueError
        If n_envs < 1.
    """
    if n_envs < 1:
        raise ValueError(f"n_envs must be >= 1, got {n_envs}")

    logger.info(
        "Creating vectorised env — n_envs=%d, seed=%s, normalize=%s",
        n_envs, seed, normalize_obs,
    )

    # Build one factory function per worker
    env_fns: List[Callable[[], CybORGWrapper]] = []
    for i in range(n_envs):
        worker_config = _worker_config(config, worker_idx=i, base_seed=seed)
        # Capture worker_config by value using default arg
        def _make(cfg=worker_config) -> CybORGWrapper:
            return CybORGWrapper(cfg)
        env_fns.append(_make)

    # Choose VecEnv class
    if vec_env_cls is None:
        vec_env_cls = DummyVecEnv if n_envs == 1 else SubprocVecEnv

    if vec_env_cls is SubprocVecEnv:
        # start_method="fork" is faster on Linux but unsafe on macOS.
        # "spawn" is safe everywhere.
        start_method = "fork" if os.name == "posix" else "spawn"
        vec_env = SubprocVecEnv(env_fns, start_method=start_method)
    else:
        vec_env = DummyVecEnv(env_fns)

    # Optionally wrap with VecMonitor (episode stats → TensorBoard)
    if monitor:
        vec_env = VecMonitor(vec_env)

    # Optionally normalise observations and rewards
    if normalize_obs:
        vec_env = VecNormalize(
            vec_env,
            norm_obs    = True,
            norm_reward = True,
            clip_obs    = 10.0,
        )

    logger.info(
        "VecEnv ready — type=%s, obs=%s, n_actions=%d",
        type(vec_env).__name__,
        vec_env.observation_space.shape,
        vec_env.action_space.n,
    )

    return vec_env


def _worker_config(
    base_config: Dict[str, Any],
    worker_idx:  int,
    base_seed:   Optional[int],
) -> Dict[str, Any]:
    """
    Build a per-worker config dict from the base config.

    Each worker gets a unique seed (base_seed + worker_idx) and
    a unique log path suffix so TensorBoard logs don't collide.
    """
    cfg = dict(base_config)

    # Assign unique seed per worker for independent episode sampling
    if base_seed is not None:
        cfg["seed"] = base_seed + worker_idx

    return cfg


class SingleEnvWrapper:
    """
    Thin wrapper that makes a single CybORGWrapper behave like a VecEnv.

    Used in evaluation and inference scripts that expect a VecEnv-style
    interface but only need one environment.

    Parameters
    ----------
    config : dict
        Environment configuration.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self._env = CybORGWrapper(config)
        self.observation_space = self._env.observation_space
        self.action_space      = self._env.action_space
        self.num_envs          = 1

    def reset(self):
        obs, info = self._env.reset()
        return obs[None]   # add batch dim

    def step(self, actions):
        action = actions[0] if hasattr(actions, "__len__") else actions
        obs, reward, terminated, truncated, info = self._env.step(int(action))
        return obs[None], [reward], [terminated or truncated], [info]

    def close(self):
        self._env.close()

    def get_network_state(self) -> Dict[str, Any]:
        return self._env.get_network_state()

    def get_metrics(self) -> Dict[str, Any]:
        return self._env.get_metrics()

    def __repr__(self) -> str:
        return f"SingleEnvWrapper({self._env!r})"