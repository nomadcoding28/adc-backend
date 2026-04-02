"""
agents/registry.py
==================
Agent factory — build any agent from a config dict without importing
concrete classes in training scripts or API handlers.

Design
------
The registry maps string names (from config.yaml ``agent_type`` field) to
factory functions.  This means:
  - Training scripts never import agent classes directly
  - New agent types can be registered without modifying existing code
  - The API can accept an ``agent_type`` string and build the right agent

Usage
-----
    # Build from config dict (primary use case)
    agent = AgentRegistry.build(env, config)

    # Register a custom agent class
    @AgentRegistry.register("my_agent")
    class MyAgent(BaseAgent): ...

    # List available agents
    print(AgentRegistry.available())

    # Check if a type is registered
    if AgentRegistry.is_registered("cvar_ppo"):
        ...

config.yaml example
-------------------
    agent:
      agent_type: "cvar_ppo"          # or "standard_ppo", "random"
      learning_rate: 3.0e-4
      total_timesteps: 2_000_000
      checkpoint_dir: "data/checkpoints"
      device: "auto"
      cvar:
        enabled: true
        alpha: 0.05
      ewc:
        enabled: true
        lambda_ewc: 0.4
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Type

from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

# Type alias for factory functions: (env, config) -> BaseAgent
AgentFactory = Callable[[Any, Dict[str, Any]], BaseAgent]


class AgentRegistry:
    """
    Central registry for all ACD agent types.

    Class-level dict maps string names → factory callables.
    Factory callables take ``(env, config)`` and return a ``BaseAgent``.

    All built-in agents are registered at the bottom of this file via
    ``_register_defaults()``.  Custom agents can be added at runtime.
    """

    _registry: Dict[str, AgentFactory] = {}

    # ------------------------------------------------------------------ #
    # Registration API
    # ------------------------------------------------------------------ #

    @classmethod
    def register(
        cls, name: str, overwrite: bool = False
    ) -> Callable[[Type[BaseAgent]], Type[BaseAgent]]:
        """
        Decorator that registers a ``BaseAgent`` subclass under ``name``.

        Parameters
        ----------
        name : str
            String key used in ``config["agent_type"]``.
        overwrite : bool
            If False (default), raise an error if the name is already taken.

        Usage
        -----
            @AgentRegistry.register("my_custom_agent")
            class MyCustomAgent(BaseAgent):
                ...
        """
        def decorator(cls_or_factory: Any) -> Any:
            if name in AgentRegistry._registry and not overwrite:
                raise ValueError(
                    f"Agent name {name!r} is already registered. "
                    f"Pass overwrite=True to replace it."
                )

            # Accept either a class (common case) or a plain factory function
            if isinstance(cls_or_factory, type) and issubclass(cls_or_factory, BaseAgent):
                # Default factory: call the class constructor
                AgentRegistry._registry[name] = lambda env, cfg: cls_or_factory(env, cfg)
            else:
                # Raw factory function
                AgentRegistry._registry[name] = cls_or_factory

            logger.debug("Registered agent type: %r", name)
            return cls_or_factory

        return decorator

    @classmethod
    def register_factory(
        cls,
        name: str,
        factory: AgentFactory,
        overwrite: bool = False,
    ) -> None:
        """
        Register a factory function directly (non-decorator form).

        Parameters
        ----------
        name : str
            String key for ``config["agent_type"]``.
        factory : callable
            Function with signature ``(env, config) -> BaseAgent``.
        overwrite : bool
            Replace existing registration if True.
        """
        if name in cls._registry and not overwrite:
            raise ValueError(
                f"Agent name {name!r} is already registered. "
                f"Pass overwrite=True to replace it."
            )
        cls._registry[name] = factory
        logger.debug("Registered agent factory: %r", name)

    # ------------------------------------------------------------------ #
    # Build API
    # ------------------------------------------------------------------ #

    @classmethod
    def build(cls, env: Any, config: Dict[str, Any]) -> BaseAgent:
        """
        Instantiate and return the agent specified in ``config["agent_type"]``.

        Parameters
        ----------
        env : Gym env
            CybORG wrapper environment.
        config : dict
            Full agent config dict.  Must contain ``"agent_type"`` key.

        Returns
        -------
        BaseAgent
            Fully initialised agent, ready for ``learn()`` or ``predict()``.

        Raises
        ------
        ValueError
            If ``agent_type`` is not registered or not present in config.

        Example
        -------
            config = load_config("config.yaml")["agent"]
            agent  = AgentRegistry.build(env, config)
        """
        agent_type = config.get("agent_type")

        if agent_type is None:
            raise ValueError(
                "config dict must contain 'agent_type' key. "
                f"Registered types: {cls.available()}"
            )

        if agent_type not in cls._registry:
            raise ValueError(
                f"Unknown agent_type {agent_type!r}. "
                f"Registered types: {cls.available()}"
            )

        logger.info("Building agent type: %r", agent_type)
        factory = cls._registry[agent_type]

        try:
            agent = factory(env, config)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to build agent {agent_type!r}: {exc}"
            ) from exc

        logger.info("Agent built successfully: %r", agent)
        return agent

    @classmethod
    def build_adversarial(
        cls,
        env: Any,
        config: Dict[str, Any],
    ) -> "AdversarialTrainer":
        """
        Build an ``AdversarialTrainer`` wrapping the agent in ``config``.

        Reads adversarial settings from ``config["adversarial"]``:
            perturbations   : list of perturbation configs
            adversarial_ratio : float, default 0.3
            reward_poison   : dict with type/rate/strategy, optional

        Parameters
        ----------
        env : Gym env
        config : dict
            Full config dict.  Agent config is read from ``config["agent"]``,
            adversarial config from ``config["adversarial"]``.

        Returns
        -------
        AdversarialTrainer
        """
        # Import here to avoid circular imports at module load
        from agents.adversarial_trainer import AdversarialTrainer
        from agents.perturbation import RewardPoisoner, build_perturbation

        agent_config = config.get("agent", config)
        agent = cls.build(env, agent_config)

        adv_config = config.get("adversarial", {})

        # Build perturbation objects from config list
        perturb_configs = adv_config.get("perturbations", [])
        obs_perturbations = [build_perturbation(dict(pc)) for pc in perturb_configs]

        # Optionally build a reward poisoner
        poison_config = adv_config.get("reward_poison")
        reward_poisoner: Optional[RewardPoisoner] = None
        if poison_config:
            reward_poisoner = RewardPoisoner(
                rate=poison_config.get("rate", 0.05),
                strategy=poison_config.get("strategy", "flip"),
                noise_std=poison_config.get("noise_std", 1.0),
            )

        return AdversarialTrainer(
            agent             = agent,
            obs_perturbations = obs_perturbations,
            reward_poisoner   = reward_poisoner,
            adversarial_ratio = adv_config.get("adversarial_ratio", 0.3),
            config            = config,
        )

    # ------------------------------------------------------------------ #
    # Query API
    # ------------------------------------------------------------------ #

    @classmethod
    def available(cls) -> List[str]:
        """Return a sorted list of all registered agent type names."""
        return sorted(cls._registry.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Return True if ``name`` is a registered agent type."""
        return name in cls._registry

    @classmethod
    def unregister(cls, name: str) -> None:
        """
        Remove an agent type from the registry.

        Primarily useful in tests that register temporary agent types.

        Parameters
        ----------
        name : str
            Name to remove.

        Raises
        ------
        KeyError
            If ``name`` is not registered.
        """
        if name not in cls._registry:
            raise KeyError(f"Agent type {name!r} is not registered.")
        del cls._registry[name]
        logger.debug("Unregistered agent type: %r", name)

    @classmethod
    def clear(cls) -> None:
        """
        Remove all registrations.

        .. warning::
            This also removes the built-in agents.  Call
            ``_register_defaults()`` afterwards to restore them.
            Only intended for test isolation.
        """
        cls._registry.clear()

    def __repr__(cls) -> str:
        return f"AgentRegistry(registered={cls.available()})"


# ══════════════════════════════════════════════════════════════════════════════
# Default agent registrations
# ══════════════════════════════════════════════════════════════════════════════

def _register_defaults() -> None:
    """
    Register all built-in agent types.

    Called once at module import time.  Uses lazy imports inside each
    factory to avoid circular import issues.
    """

    # ── CVaR-PPO (primary agent — Novelty 3) ──────────────────────────────
    def _build_cvar_ppo(env: Any, config: Dict[str, Any]) -> BaseAgent:
        from agents.cvar_ppo import ACDPPOAgent
        return ACDPPOAgent(env, config)

    AgentRegistry.register_factory("cvar_ppo", _build_cvar_ppo)

    # ── Standard PPO (ablation baseline) ──────────────────────────────────
    def _build_standard_ppo(env: Any, config: Dict[str, Any]) -> BaseAgent:
        from agents.cvar_ppo import ACDPPOAgent
        # Disable CVaR by overriding the config
        cfg = dict(config)
        cfg.setdefault("cvar", {})["enabled"] = False
        cfg.setdefault("ewc", {})["enabled"] = False
        return ACDPPOAgent(env, cfg)

    AgentRegistry.register_factory("standard_ppo", _build_standard_ppo)

    # ── PPO + CVaR only (no EWC ablation) ─────────────────────────────────
    def _build_cvar_no_ewc(env: Any, config: Dict[str, Any]) -> BaseAgent:
        from agents.cvar_ppo import ACDPPOAgent
        cfg = dict(config)
        cfg.setdefault("cvar", {})["enabled"] = True
        cfg.setdefault("ewc", {})["enabled"] = False
        return ACDPPOAgent(env, cfg)

    AgentRegistry.register_factory("cvar_ppo_no_ewc", _build_cvar_no_ewc)

    # ── PPO + EWC only (no CVaR ablation) ─────────────────────────────────
    def _build_ppo_ewc_only(env: Any, config: Dict[str, Any]) -> BaseAgent:
        from agents.cvar_ppo import ACDPPOAgent
        cfg = dict(config)
        cfg.setdefault("cvar", {})["enabled"] = False
        cfg.setdefault("ewc", {})["enabled"] = True
        return ACDPPOAgent(env, cfg)

    AgentRegistry.register_factory("ppo_ewc_only", _build_ppo_ewc_only)

    # ── Random agent (lower-bound baseline) ───────────────────────────────
    def _build_random(env: Any, config: Dict[str, Any]) -> BaseAgent:
        return _RandomAgent(env, config)

    AgentRegistry.register_factory("random", _build_random)

    logger.debug(
        "Registered built-in agent types: %s", AgentRegistry.available()
    )


# ══════════════════════════════════════════════════════════════════════════════
# Random agent — lower-bound baseline
# ══════════════════════════════════════════════════════════════════════════════

class _RandomAgent(BaseAgent):
    """
    Random action agent — used as the absolute lower-bound baseline.

    Takes uniformly random actions from the action space.
    No learning, no parameters.  Used in the benchmark table.
    """

    def learn(self, total_timesteps: int, **kwargs: Any) -> "_RandomAgent":
        # Random agent does not learn — but we count the steps
        self._total_timesteps_trained += total_timesteps
        return self

    def predict(
        self,
        observation: Any,
        deterministic: bool = True,
    ) -> tuple:
        action = self.env.action_space.sample()
        import numpy as np
        return np.array([action]), None

    def save(self, path: Any) -> Any:
        import json, pathlib
        p = pathlib.Path(str(path) + ".json")
        p.write_text(json.dumps({"agent_type": "random"}))
        return p

    @classmethod
    def load(cls, path: Any, env: Any, config: Dict[str, Any]) -> "_RandomAgent":
        return cls(env, config)

    def get_metrics(self) -> Dict[str, Any]:
        return {
            **self.get_base_metrics(),
            "mean_reward": None,
            "note":        "Random agent — no learning",
        }


# ── Register defaults immediately on import ───────────────────────────────
_register_defaults()