"""
agents/base_agent.py
====================
Abstract base class that every ACD agent must implement.

Design principles
-----------------
- Forces a consistent interface across CVaR-PPO, standard PPO, and any
  future agent variants so the API layer and training scripts never need
  to know which concrete agent they're dealing with.
- Provides shared utility methods (device resolution, checkpoint paths,
  config validation) so subclasses don't duplicate boilerplate.
- Uses Python ABCs so missing method implementations raise at import time,
  not at runtime during a long training run.

Usage
-----
    class MyAgent(BaseAgent):
        def learn(self, total_timesteps, **kwargs): ...
        def predict(self, observation, deterministic): ...
        def save(self, path): ...
        def load(cls, path, env, config): ...
        def get_metrics(self): ...
"""

from __future__ import annotations

import abc
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch


class BaseAgent(abc.ABC):
    """
    Abstract base for all ACD reinforcement learning agents.

    Every concrete agent (CVaRPPO, standard PPO, random baseline, etc.)
    must implement the five abstract methods defined here.  The base class
    provides shared state (config, device, checkpoint directory) and a
    handful of concrete utility helpers.

    Parameters
    ----------
    env:
        A Gym-compatible environment.  In practice this will always be
        an instance of ``envs.CybORGWrapper``.
    config : dict
        Full agent configuration dict (loaded from config.yaml).
        Required keys are validated in ``_validate_config``.
    """

    # ------------------------------------------------------------------ #
    # Required config keys every agent must supply
    # ------------------------------------------------------------------ #
    _REQUIRED_CONFIG_KEYS: Tuple[str, ...] = (
        "agent_type",
        "learning_rate",
        "total_timesteps",
        "checkpoint_dir",
    )

    def __init__(self, env: Any, config: Dict[str, Any]) -> None:
        self.env = env
        self.config = config

        self._validate_config()

        # Resolve compute device
        self.device: torch.device = self._resolve_device(
            config.get("device", "auto")
        )

        # Checkpoint directory
        self.checkpoint_dir = Path(config["checkpoint_dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Bookkeeping
        self._training_start_time: Optional[float] = None
        self._total_timesteps_trained: int = 0
        self._episode_count: int = 0
        self._is_training: bool = False

    # ------------------------------------------------------------------ #
    # Abstract interface — every subclass must implement these
    # ------------------------------------------------------------------ #

    @abc.abstractmethod
    def learn(
        self,
        total_timesteps: int,
        reset_num_timesteps: bool = True,
        **kwargs: Any,
    ) -> "BaseAgent":
        """
        Run the training loop for ``total_timesteps`` environment steps.

        Parameters
        ----------
        total_timesteps : int
            Number of environment steps to train for.
        reset_num_timesteps : bool
            If False, continue counting from the current step (used for
            incremental adaptation after drift).

        Returns
        -------
        BaseAgent
            Returns ``self`` to allow method chaining.
        """

    @abc.abstractmethod
    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
    ) -> Tuple[np.ndarray, Optional[Any]]:
        """
        Select an action given an observation.

        Parameters
        ----------
        observation : np.ndarray
            Current environment observation vector.
        deterministic : bool
            If True, select the greedy (most probable) action.
            If False, sample from the policy distribution (used during
            training rollout collection).

        Returns
        -------
        action : np.ndarray
            Selected action.
        state : Any or None
            Recurrent state (None for non-recurrent policies).
        """

    @abc.abstractmethod
    def save(self, path: Union[str, Path]) -> Path:
        """
        Persist the agent's weights and config to disk.

        Parameters
        ----------
        path : str or Path
            Directory or file path prefix for the saved checkpoint.

        Returns
        -------
        Path
            Resolved path where the checkpoint was written.
        """

    @classmethod
    @abc.abstractmethod
    def load(
        cls,
        path: Union[str, Path],
        env: Any,
        config: Dict[str, Any],
    ) -> "BaseAgent":
        """
        Restore an agent from a saved checkpoint.

        Parameters
        ----------
        path : str or Path
            Path to the saved checkpoint (as returned by ``save``).
        env : Gym env
            Environment to attach to the loaded agent.
        config : dict
            Config dict — may override values stored in the checkpoint.

        Returns
        -------
        BaseAgent
            Fully restored agent ready for inference or continued training.
        """

    @abc.abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """
        Return a snapshot of the agent's current training metrics.

        Returns
        -------
        dict
            Keys are metric names (strings), values are scalars or lists.
            This dict is serialised and sent to the frontend via the API.

        Example return value
        --------------------
        {
            "mean_reward":      8.74,
            "cvar_005":        -2.14,
            "total_timesteps":  1_247_892,
            "episode_count":    4_821,
            "loss_policy":      0.042,
            "loss_value":       0.118,
            "loss_ewc":         0.081,
        }
        """

    # ------------------------------------------------------------------ #
    # Concrete helpers shared by all agents
    # ------------------------------------------------------------------ #

    def start_training(self) -> None:
        """Mark the agent as currently training and record the start time."""
        self._is_training = True
        self._training_start_time = time.monotonic()

    def stop_training(self) -> None:
        """Mark the agent as not training."""
        self._is_training = False

    @property
    def is_training(self) -> bool:
        """True while a ``learn()`` call is in progress."""
        return self._is_training

    @property
    def total_timesteps_trained(self) -> int:
        """Total environment steps completed across all ``learn()`` calls."""
        return self._total_timesteps_trained

    @property
    def episode_count(self) -> int:
        """Total episodes completed across all ``learn()`` calls."""
        return self._episode_count

    def training_elapsed(self) -> float:
        """Seconds since the current training run started (0 if not training)."""
        if self._training_start_time is None:
            return 0.0
        return time.monotonic() - self._training_start_time

    def save_checkpoint(self, tag: str = "") -> Path:
        """
        Convenience wrapper: save to the configured checkpoint directory.

        Parameters
        ----------
        tag : str
            Optional suffix appended to the filename, e.g. ``"best"`` or
            ``"ep4821"``.

        Returns
        -------
        Path
            Path to the saved checkpoint file.
        """
        suffix = f"_{tag}" if tag else ""
        filename = f"{self.config['agent_type']}{suffix}"
        path = self.checkpoint_dir / filename
        return self.save(path)

    def get_base_metrics(self) -> Dict[str, Any]:
        """
        Return metrics that are common to every agent type.

        Subclasses should call ``super().get_base_metrics()`` and merge
        their own metrics on top.
        """
        return {
            "agent_type":          self.config["agent_type"],
            "device":              str(self.device),
            "total_timesteps":     self._total_timesteps_trained,
            "episode_count":       self._episode_count,
            "is_training":         self._is_training,
            "training_elapsed_s":  round(self.training_elapsed(), 2),
            "checkpoint_dir":      str(self.checkpoint_dir),
        }

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _validate_config(self) -> None:
        """
        Raise ``ValueError`` if any required config key is missing.

        Called during ``__init__`` so problems surface immediately rather
        than mid-training.
        """
        missing = [k for k in self._REQUIRED_CONFIG_KEYS if k not in self.config]
        if missing:
            raise ValueError(
                f"Agent config is missing required keys: {missing}. "
                f"Check your config.yaml."
            )

    @staticmethod
    def _resolve_device(device_str: str) -> torch.device:
        """
        Resolve the device string to a ``torch.device``.

        Accepts ``"auto"``, ``"cuda"``, ``"cuda:0"``, ``"mps"``, ``"cpu"``.
        ``"auto"`` picks CUDA if available, then MPS (Apple Silicon), then CPU.
        """
        if device_str == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(device_str)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"agent_type={self.config.get('agent_type', 'unknown')!r}, "
            f"device={self.device}, "
            f"timesteps={self._total_timesteps_trained:,})"
        )