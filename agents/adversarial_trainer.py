"""
agents/adversarial_trainer.py
==============================
Adversarial min-max training loop — Novelty 2 of the ACD Framework.

Overview
--------
Standard RL trains an agent on clean environment observations.  A real
adversary, however, can:
  1. Spoof sensor readings (observation perturbation)
  2. Poison the reward signal (corrupted feedback)
  3. Delay telemetry (stale observations)

The ``AdversarialTrainer`` wraps an ``ACDPPOAgent`` and interleaves
*clean episodes* with *adversarial episodes* at a configurable ratio.
During adversarial episodes, observations are perturbed using FGSM or PGD
before being passed to the agent.

Min-Max Objective
-----------------
    max_θ min_δ E[R(s + δ)]

The inner minimisation (finding δ that reduces reward) is approximated
by a single FGSM/PGD step each time an adversarial episode is triggered.
The outer maximisation is the standard PPO update.

Training protocol
-----------------
    for each rollout:
        if random() < adversarial_ratio:
            perturb observation with FGSM/PGD
            collect adversarial rollout
        else:
            collect clean rollout
        update policy with CVaR-PPO

Usage
-----
    from agents import ACDPPOAgent, AdversarialTrainer
    from agents.perturbation import FGSMPerturbation, RewardPoisoner

    agent  = ACDPPOAgent(env, config)
    fgsm   = FGSMPerturbation(epsilon=0.1)
    pgd    = PGDPerturbation(epsilon=0.1, steps=10)
    poison = RewardPoisoner(rate=0.05, strategy="flip")

    trainer = AdversarialTrainer(
        agent            = agent,
        obs_perturbations = [fgsm, pgd],
        reward_poisoner   = poison,
        adversarial_ratio = 0.3,
        config            = config,
    )
    trainer.train(total_timesteps=1_000_000)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback

from agents.cvar_ppo import ACDPPOAgent
from agents.perturbation import BasePerturbation, RewardPoisoner

logger = logging.getLogger(__name__)


class AdversarialCallback(BaseCallback):
    """
    SB3 callback that injects adversarial perturbations at the rollout level.

    This hooks into ``_on_rollout_start`` to decide whether the upcoming
    rollout is clean or adversarial, and into ``_on_step`` to perturb
    observations and rewards on adversarial rollouts.

    Parameters
    ----------
    obs_perturbations : list[BasePerturbation]
        Pool of perturbation objects.  One is randomly chosen per
        adversarial rollout.
    reward_poisoner : RewardPoisoner, optional
        If provided, poisons rewards on adversarial rollouts as well.
    adversarial_ratio : float
        Fraction of rollouts that will be adversarial.
    on_adversarial_step : callable, optional
        Optional hook called after each adversarial step for monitoring.
    """

    def __init__(
        self,
        obs_perturbations: List[BasePerturbation],
        reward_poisoner: Optional[RewardPoisoner] = None,
        adversarial_ratio: float = 0.3,
        on_adversarial_step: Optional[Callable] = None,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.obs_perturbations = obs_perturbations
        self.reward_poisoner = reward_poisoner
        self.adversarial_ratio = adversarial_ratio
        self.on_adversarial_step = on_adversarial_step

        # State
        self._is_adversarial_rollout: bool = False
        self._current_perturbation: Optional[BasePerturbation] = None

        # Metrics
        self._adversarial_steps: int = 0
        self._clean_steps: int = 0
        self._adversarial_rewards: List[float] = []
        self._clean_rewards: List[float] = []

    def _on_rollout_start(self) -> None:
        """Decide at the start of each rollout whether it will be adversarial."""
        self._is_adversarial_rollout = (
            np.random.random() < self.adversarial_ratio
            and len(self.obs_perturbations) > 0
        )

        if self._is_adversarial_rollout:
            # Pick a random perturbation type for this rollout
            idx = np.random.randint(0, len(self.obs_perturbations))
            self._current_perturbation = self.obs_perturbations[idx]
            if self.verbose >= 1:
                logger.debug(
                    "Adversarial rollout — using %s",
                    self._current_perturbation.__class__.__name__,
                )
        else:
            self._current_perturbation = None

    def _on_step(self) -> bool:
        """
        Called after every environment step.

        On adversarial rollouts, we retroactively perturb the observation
        that was just stored in the rollout buffer.  This is the cleanest
        way to inject perturbations without modifying the SB3 collect loop.
        """
        if not self._is_adversarial_rollout or self._current_perturbation is None:
            self._clean_steps += 1
            reward = self.locals.get("rewards", [0])[0]
            self._clean_rewards.append(float(reward))
            return True

        self._adversarial_steps += 1

        # ------------------------------------------------------------------
        # Perturb the most recently stored observation in the rollout buffer
        # ------------------------------------------------------------------
        model = self.model
        if hasattr(model, "rollout_buffer") and model.rollout_buffer.pos > 0:
            buf = model.rollout_buffer
            idx = (buf.pos - 1) % buf.buffer_size

            obs_tensor = torch.tensor(
                buf.observations[idx],
                dtype=torch.float32,
                device=model.device,
            )

            # Get the action that was taken for gradient computation
            action_tensor = torch.tensor(
                buf.actions[idx],
                dtype=torch.long,
                device=model.device,
            )

            with torch.no_grad():
                # Only FGSM/PGD need the policy — noise-based ones don't
                try:
                    obs_perturbed = self._current_perturbation.perturb(
                        obs_tensor, policy=model.policy, actions=action_tensor
                    )
                except TypeError:
                    obs_perturbed = self._current_perturbation.perturb(obs_tensor)

            buf.observations[idx] = obs_perturbed.cpu().numpy()

        # ------------------------------------------------------------------
        # Optionally poison the reward
        # ------------------------------------------------------------------
        reward = self.locals.get("rewards", [0])[0]
        if self.reward_poisoner is not None:
            reward = self.reward_poisoner.poison(float(reward))

        self._adversarial_rewards.append(float(reward))

        if self.on_adversarial_step is not None:
            self.on_adversarial_step(
                step=self.num_timesteps,
                perturbation=self._current_perturbation.__class__.__name__,
                reward=reward,
            )

        return True

    def get_metrics(self) -> Dict[str, Any]:
        """Return adversarial training statistics."""
        total = self._adversarial_steps + self._clean_steps
        adv_mean = (
            float(np.mean(self._adversarial_rewards))
            if self._adversarial_rewards else 0.0
        )
        clean_mean = (
            float(np.mean(self._clean_rewards))
            if self._clean_rewards else 0.0
        )
        return {
            "adversarial_steps":       self._adversarial_steps,
            "clean_steps":             self._clean_steps,
            "total_steps":             total,
            "observed_adv_ratio":      round(self._adversarial_steps / max(total, 1), 4),
            "mean_reward_adversarial": round(adv_mean, 4),
            "mean_reward_clean":       round(clean_mean, 4),
            "robustness_ratio":        round(adv_mean / max(abs(clean_mean), 1e-6), 4),
        }


class AdversarialTrainer:
    """
    Orchestrates adversarial min-max training for the ACD Framework.

    Wraps an ``ACDPPOAgent`` and adds adversarial perturbations at the
    rollout level.  The underlying CVaR-PPO + EWC training is unchanged —
    this class only controls *which* observations the agent trains on.

    Parameters
    ----------
    agent : ACDPPOAgent
        The agent to train adversarially.
    obs_perturbations : list[BasePerturbation]
        Observation perturbation methods to use (FGSM, PGD, noise, delay).
        Can be a single perturbation or a mix — one is sampled per rollout.
    reward_poisoner : RewardPoisoner, optional
        If provided, poisons a fraction of rewards during adversarial rollouts.
    adversarial_ratio : float
        Fraction of rollouts that will be adversarial.  Default 0.3 (30%).
    config : dict
        Full training config dict.

    Examples
    --------
    Basic usage::

        fgsm    = FGSMPerturbation(epsilon=0.1)
        pgd     = PGDPerturbation(epsilon=0.1, steps=10)
        noise   = GaussianNoisePerturbation(std=0.05)
        poisoner = RewardPoisoner(rate=0.05)

        trainer = AdversarialTrainer(
            agent             = agent,
            obs_perturbations = [fgsm, pgd, noise],
            reward_poisoner   = poisoner,
            adversarial_ratio = 0.3,
            config            = config,
        )
        trainer.train(total_timesteps=1_000_000)

    With custom monitoring callback::

        trainer.train(
            total_timesteps = 1_000_000,
            extra_callbacks = [my_tensorboard_callback],
        )
    """

    def __init__(
        self,
        agent: ACDPPOAgent,
        obs_perturbations: List[BasePerturbation],
        reward_poisoner: Optional[RewardPoisoner] = None,
        adversarial_ratio: float = 0.3,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not 0 <= adversarial_ratio <= 1:
            raise ValueError(
                f"adversarial_ratio must be in [0, 1], got {adversarial_ratio}"
            )
        if not obs_perturbations:
            logger.warning(
                "AdversarialTrainer created with no perturbations — "
                "training will be equivalent to clean training."
            )

        self.agent = agent
        self.obs_perturbations = obs_perturbations
        self.reward_poisoner = reward_poisoner
        self.adversarial_ratio = adversarial_ratio
        self.config = config or {}

        # Build the adversarial callback
        self._adv_callback = AdversarialCallback(
            obs_perturbations = obs_perturbations,
            reward_poisoner   = reward_poisoner,
            adversarial_ratio = adversarial_ratio,
            verbose           = self.config.get("verbose", 0),
        )

        # Training history
        self._training_runs: List[Dict[str, Any]] = []

    def train(
        self,
        total_timesteps: int,
        reset_num_timesteps: bool = True,
        extra_callbacks: Optional[List[BaseCallback]] = None,
        checkpoint_freq: int = 50_000,
    ) -> Dict[str, Any]:
        """
        Run adversarial training for ``total_timesteps`` steps.

        Parameters
        ----------
        total_timesteps : int
            Total environment steps to train for.
        reset_num_timesteps : bool
            If False, continue from current step count (post-drift adaptation).
        extra_callbacks : list[BaseCallback], optional
            Additional SB3 callbacks (e.g. TensorBoard, checkpointing).
        checkpoint_freq : int
            Save a checkpoint every this many timesteps.

        Returns
        -------
        dict
            Training result metrics.
        """
        logger.info(
            "Starting adversarial training — "
            "total_timesteps=%d, adversarial_ratio=%.1f%%, "
            "perturbations=%s",
            total_timesteps,
            self.adversarial_ratio * 100,
            [p.__class__.__name__ for p in self.obs_perturbations],
        )

        start_time = time.monotonic()

        # Combine adversarial callback with any user-supplied callbacks
        all_callbacks: List[BaseCallback] = [self._adv_callback]
        if extra_callbacks:
            all_callbacks.extend(extra_callbacks)

        # Delegate actual training to the agent's learn() method
        # The adversarial callback hooks in at the SB3 level
        self.agent.learn(
            total_timesteps     = total_timesteps,
            reset_num_timesteps = reset_num_timesteps,
            callback            = all_callbacks,
        )

        elapsed = time.monotonic() - start_time
        adv_metrics = self._adv_callback.get_metrics()
        agent_metrics = self.agent.get_metrics()

        result = {
            "total_timesteps":    total_timesteps,
            "elapsed_s":          round(elapsed, 2),
            "steps_per_sec":      round(total_timesteps / max(elapsed, 1), 1),
            **adv_metrics,
            **{f"agent_{k}": v for k, v in agent_metrics.items()},
        }

        self._training_runs.append(result)

        logger.info(
            "Adversarial training complete — "
            "elapsed=%.1fs, clean_reward=%.3f, adv_reward=%.3f, "
            "robustness_ratio=%.3f",
            elapsed,
            adv_metrics["mean_reward_clean"],
            adv_metrics["mean_reward_adversarial"],
            adv_metrics["robustness_ratio"],
        )

        return result

    # ---------------------------------------------------------------------- #
    # Evaluation
    # ---------------------------------------------------------------------- #

    def evaluate_robustness(
        self,
        n_episodes: int = 50,
        perturbation: Optional[BasePerturbation] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate the agent's robustness under a given perturbation.

        Runs ``n_episodes`` episodes with clean observations and
        ``n_episodes`` with perturbed observations, then reports the
        reward ratio (adversarial / clean) as the robustness score.

        Parameters
        ----------
        n_episodes : int
            Number of episodes for each condition.
        perturbation : BasePerturbation, optional
            The perturbation to evaluate against.  If None, uses the
            first perturbation in ``self.obs_perturbations``.

        Returns
        -------
        dict
            Keys: ``clean_reward``, ``adv_reward``, ``robustness``,
            ``perturbation_type``, ``n_episodes``.
        """
        if perturbation is None:
            if not self.obs_perturbations:
                raise ValueError("No perturbations configured for evaluation.")
            perturbation = self.obs_perturbations[0]

        env = self.agent.env

        def _run_episodes(perturb: bool) -> Tuple[float, List[float]]:
            rewards = []
            for _ in range(n_episodes):
                obs, _ = env.reset()
                ep_reward = 0.0
                done = False
                while not done:
                    if perturb:
                        obs_t = torch.tensor(obs, dtype=torch.float32)
                        obs_t = perturbation.perturb(obs_t)
                        obs = obs_t.numpy()

                    action, _ = self.agent.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = env.step(action)
                    ep_reward += reward
                    done = terminated or truncated

                rewards.append(ep_reward)

            return float(np.mean(rewards)), rewards

        clean_mean, clean_rewards = _run_episodes(perturb=False)
        adv_mean, adv_rewards = _run_episodes(perturb=True)

        robustness = adv_mean / max(abs(clean_mean), 1e-6)

        return {
            "perturbation_type": perturbation.__class__.__name__,
            "perturbation_config": perturbation.get_config(),
            "n_episodes":        n_episodes,
            "clean_reward_mean": round(clean_mean, 4),
            "clean_reward_std":  round(float(np.std(clean_rewards)), 4),
            "adv_reward_mean":   round(adv_mean, 4),
            "adv_reward_std":    round(float(np.std(adv_rewards)), 4),
            "robustness":        round(float(robustness), 4),
            "reward_gap":        round(clean_mean - adv_mean, 4),
        }

    def evaluate_all_perturbations(
        self, n_episodes: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Run robustness evaluation for every configured perturbation.

        Returns a list of result dicts, one per perturbation type.
        Useful for generating the robustness table in the paper / dashboard.
        """
        results = []
        for perturb in self.obs_perturbations:
            logger.info("Evaluating robustness against %s", perturb.__class__.__name__)
            result = self.evaluate_robustness(
                n_episodes=n_episodes, perturbation=perturb
            )
            results.append(result)
        return results

    # ---------------------------------------------------------------------- #
    # Metrics / state
    # ---------------------------------------------------------------------- #

    def get_metrics(self) -> Dict[str, Any]:
        """
        Return current adversarial training metrics for the API / dashboard.
        """
        adv = self._adv_callback.get_metrics()
        poison_cfg = (
            self.reward_poisoner.get_config()
            if self.reward_poisoner else None
        )
        return {
            "adversarial_ratio":        self.adversarial_ratio,
            "n_perturbation_types":     len(self.obs_perturbations),
            "perturbation_types":       [p.__class__.__name__ for p in self.obs_perturbations],
            "reward_poisoner":          poison_cfg,
            "n_training_runs":          len(self._training_runs),
            **adv,
        }

    def save_checkpoint(self, tag: str = "adversarial") -> Path:
        """Save the underlying agent's checkpoint."""
        return self.agent.save_checkpoint(tag=tag)

    def __repr__(self) -> str:
        return (
            f"AdversarialTrainer("
            f"ratio={self.adversarial_ratio:.0%}, "
            f"perturbations={[p.__class__.__name__ for p in self.obs_perturbations]}, "
            f"agent={self.agent.__class__.__name__})"
        )