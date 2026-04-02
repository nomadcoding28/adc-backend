"""
agents/perturbation.py
======================
Adversarial perturbation generators for robustness training (Novelty 2).

Each class produces a *perturbed copy* of an input tensor while leaving
the original untouched.  They are designed to be composable and
configurable so the ``AdversarialTrainer`` can mix and match them.

Classes
-------
    BasePerturbation              Abstract base — all perturbations inherit this
    FGSMPerturbation              Fast Gradient Sign Method (single step)
    PGDPerturbation               Projected Gradient Descent (iterative)
    GaussianNoisePerturbation     Additive i.i.d. Gaussian noise
    RewardPoisoner                Randomly flips / zeroes reward signals
    ObservationDelayPerturbation  Simulates stale / delayed observations

Mathematical background
-----------------------
FGSM (Goodfellow et al., 2014):
    x_adv = x + ε · sign(∇_x L(θ, x, y))

PGD (Madry et al., 2018):
    x_0   = x + U(-ε, ε)            # random start inside ε-ball
    x_t+1 = Π_{x+S}(x_t + α · sign(∇_x L(θ, x_t, y)))

where Π projects back into the ε-ball and the observation bounds.

Usage
-----
    fgsm = FGSMPerturbation(epsilon=0.1)
    x_adv = fgsm.perturb(obs_tensor, policy_model)

    pgd = PGDPerturbation(epsilon=0.1, steps=10, alpha=0.01)
    x_adv = pgd.perturb(obs_tensor, policy_model)

    noise = GaussianNoisePerturbation(std=0.05)
    x_noisy = noise.perturb(obs_tensor)
"""

from __future__ import annotations

import abc
import random
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# ══════════════════════════════════════════════════════════════════════════════
# Base
# ══════════════════════════════════════════════════════════════════════════════

class BasePerturbation(abc.ABC):
    """
    Abstract base class for all adversarial perturbations.

    Subclasses must implement ``perturb``.  All perturbations are
    stateless — the same instance can be reused across batches.
    """

    def __init__(self, clip_min: float = 0.0, clip_max: float = 1.0) -> None:
        """
        Parameters
        ----------
        clip_min, clip_max : float
            Valid range for observation values.  Perturbed outputs are
            clipped to [clip_min, clip_max] to keep them in the
            observation space.  Defaults assume normalised observations.
        """
        self.clip_min = clip_min
        self.clip_max = clip_max

    @abc.abstractmethod
    def perturb(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """
        Return a perturbed copy of ``x``.

        Parameters
        ----------
        x : torch.Tensor
            Input observation tensor, shape (batch, obs_dim) or (obs_dim,).

        Returns
        -------
        torch.Tensor
            Adversarially perturbed tensor, same shape as input.
        """

    def _clip(self, x: torch.Tensor) -> torch.Tensor:
        """Clip tensor values into the valid observation range."""
        return x.clamp(self.clip_min, self.clip_max)

    def get_config(self) -> Dict[str, Any]:
        """Return a JSON-serialisable config dict for logging."""
        return {
            "type":     self.__class__.__name__,
            "clip_min": self.clip_min,
            "clip_max": self.clip_max,
        }

    def __repr__(self) -> str:
        cfg = self.get_config()
        params = ", ".join(f"{k}={v}" for k, v in cfg.items() if k != "type")
        return f"{self.__class__.__name__}({params})"


# ══════════════════════════════════════════════════════════════════════════════
# FGSM — Fast Gradient Sign Method
# ══════════════════════════════════════════════════════════════════════════════

class FGSMPerturbation(BasePerturbation):
    """
    Fast Gradient Sign Method (Goodfellow et al., 2014).

    Adds a single-step perturbation in the direction of the gradient of the
    policy loss with respect to the observation:

        x_adv = clip(x + ε · sign(∇_x L), clip_min, clip_max)

    This is the fastest and most common adversarial attack, and acts as the
    lower bound for robustness evaluation.

    Parameters
    ----------
    epsilon : float
        Perturbation budget.  Typical values: 0.05, 0.1, 0.2.
    """

    def __init__(
        self,
        epsilon: float = 0.1,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
    ) -> None:
        super().__init__(clip_min, clip_max)
        if not 0 < epsilon <= 1:
            raise ValueError(f"epsilon must be in (0, 1], got {epsilon}")
        self.epsilon = epsilon

    def perturb(
        self,
        x: torch.Tensor,
        policy: Optional[nn.Module] = None,
        actions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply FGSM perturbation.

        Parameters
        ----------
        x : torch.Tensor
            Observation tensor, shape (batch, obs_dim).  Must have
            ``requires_grad=True`` or will be converted internally.
        policy : nn.Module, optional
            Policy network.  If provided, uses the policy's log-prob loss
            as the gradient source.  If None, uses a sign-random direction
            (useful for testing without a policy).
        actions : torch.Tensor, optional
            Actions corresponding to ``x``.  Required when policy is given.

        Returns
        -------
        torch.Tensor
            Perturbed observation tensor, same shape as ``x``.
        """
        x = x.detach().clone()

        if policy is not None and actions is not None:
            x.requires_grad_(True)

            # Forward pass through policy to get action log-probs
            try:
                # SB3-style policy: evaluate_actions returns (values, log_prob, entropy)
                _, log_prob, _ = policy.evaluate_actions(x, actions)
                loss = -log_prob.mean()       # maximise loss = minimise log-prob
            except AttributeError:
                # Fallback: treat policy as a plain nn.Module
                logits = policy(x)
                loss = -torch.log_softmax(logits, dim=-1).gather(
                    1, actions.unsqueeze(1)
                ).mean()

            policy.zero_grad()
            loss.backward()

            grad_sign = x.grad.sign()
            x_adv = x.detach() + self.epsilon * grad_sign
        else:
            # No policy available: use a random sign direction (ablation / testing)
            grad_sign = torch.randint_like(x, low=0, high=2).float() * 2 - 1
            x_adv = x + self.epsilon * grad_sign

        return self._clip(x_adv)

    def get_config(self) -> Dict[str, Any]:
        cfg = super().get_config()
        cfg["epsilon"] = self.epsilon
        return cfg


# ══════════════════════════════════════════════════════════════════════════════
# PGD — Projected Gradient Descent
# ══════════════════════════════════════════════════════════════════════════════

class PGDPerturbation(BasePerturbation):
    """
    Projected Gradient Descent attack (Madry et al., 2018).

    Iteratively applies FGSM-style steps with projection back to the
    ε-ball, making it significantly stronger than single-step FGSM:

        x_0   = x + U(-ε, ε)                  (random initialisation)
        x_t+1 = clip(x_t + α · sign(∇_x L), x-ε, x+ε)

    Parameters
    ----------
    epsilon : float
        Total perturbation budget (L∞ ball radius).
    steps : int
        Number of gradient steps (default 10).
    alpha : float
        Step size per iteration.  Rule of thumb: ε / steps * 2.
    random_start : bool
        If True, initialise with a random perturbation inside the ε-ball.
        Almost always True in practice.
    """

    def __init__(
        self,
        epsilon: float = 0.1,
        steps: int = 10,
        alpha: Optional[float] = None,
        random_start: bool = True,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
    ) -> None:
        super().__init__(clip_min, clip_max)
        self.epsilon = epsilon
        self.steps = steps
        self.alpha = alpha if alpha is not None else epsilon / steps * 2.0
        self.random_start = random_start

    def perturb(
        self,
        x: torch.Tensor,
        policy: Optional[nn.Module] = None,
        actions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply PGD attack.

        Parameters
        ----------
        x : torch.Tensor
            Clean observation tensor, shape (batch, obs_dim).
        policy : nn.Module, optional
            Policy network for gradient computation.
        actions : torch.Tensor, optional
            Corresponding actions.

        Returns
        -------
        torch.Tensor
            Strongly adversarially perturbed observation.
        """
        x_orig = x.detach().clone()

        # Random initialisation within ε-ball
        if self.random_start:
            delta = torch.zeros_like(x_orig).uniform_(-self.epsilon, self.epsilon)
            x_adv = self._clip(x_orig + delta)
        else:
            x_adv = x_orig.clone()

        for _ in range(self.steps):
            x_adv = x_adv.detach().clone().requires_grad_(True)

            if policy is not None and actions is not None:
                try:
                    _, log_prob, _ = policy.evaluate_actions(x_adv, actions)
                    loss = -log_prob.mean()
                except AttributeError:
                    logits = policy(x_adv)
                    loss = -torch.log_softmax(logits, dim=-1).gather(
                        1, actions.unsqueeze(1)
                    ).mean()

                if policy.optimizer is not None:
                    policy.optimizer.zero_grad()
                loss.backward()
                grad_sign = x_adv.grad.sign()
            else:
                grad_sign = torch.randint_like(x_adv, low=0, high=2).float() * 2 - 1

            # Gradient step
            x_adv = x_adv.detach() + self.alpha * grad_sign

            # Project back into ε-ball around original observation
            delta = torch.clamp(x_adv - x_orig, -self.epsilon, self.epsilon)
            x_adv = self._clip(x_orig + delta)

        return x_adv.detach()

    def get_config(self) -> Dict[str, Any]:
        cfg = super().get_config()
        cfg.update({
            "epsilon":      self.epsilon,
            "steps":        self.steps,
            "alpha":        self.alpha,
            "random_start": self.random_start,
        })
        return cfg


# ══════════════════════════════════════════════════════════════════════════════
# Gaussian Noise
# ══════════════════════════════════════════════════════════════════════════════

class GaussianNoisePerturbation(BasePerturbation):
    """
    Additive i.i.d. Gaussian noise perturbation.

    Models sensor measurement error and low-level telemetry noise.
    Does not require gradient computation — usable without a policy.

        x_adv = clip(x + N(0, std²), clip_min, clip_max)

    Parameters
    ----------
    std : float
        Standard deviation of the Gaussian noise.
    """

    def __init__(
        self,
        std: float = 0.05,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
    ) -> None:
        super().__init__(clip_min, clip_max)
        if std <= 0:
            raise ValueError(f"std must be positive, got {std}")
        self.std = std

    def perturb(self, x: torch.Tensor, **_: Any) -> torch.Tensor:
        """
        Add Gaussian noise to ``x``.

        Parameters
        ----------
        x : torch.Tensor
            Input observation tensor.

        Returns
        -------
        torch.Tensor
            Noisy observation tensor.
        """
        noise = torch.randn_like(x) * self.std
        return self._clip(x + noise)

    def get_config(self) -> Dict[str, Any]:
        cfg = super().get_config()
        cfg["std"] = self.std
        return cfg


# ══════════════════════════════════════════════════════════════════════════════
# Reward Poisoner
# ══════════════════════════════════════════════════════════════════════════════

class RewardPoisoner:
    """
    Adversarially corrupts reward signals during training.

    Models a scenario where an attacker has partial control over the
    defender's feedback loop (e.g., via compromised IDS sensors).

    Two poisoning strategies are supported:
        "flip"  — flip the sign of the reward with probability ``rate``
        "zero"  — set the reward to zero with probability ``rate``
        "noise" — add Gaussian noise to the reward

    Parameters
    ----------
    rate : float
        Fraction of rewards to poison, in [0, 1].  Typical: 0.05.
    strategy : str
        One of ``"flip"``, ``"zero"``, ``"noise"``.
    noise_std : float
        Standard deviation for ``"noise"`` strategy.  Ignored otherwise.
    """

    STRATEGIES = ("flip", "zero", "noise")

    def __init__(
        self,
        rate: float = 0.05,
        strategy: str = "flip",
        noise_std: float = 1.0,
    ) -> None:
        if not 0 <= rate <= 1:
            raise ValueError(f"rate must be in [0, 1], got {rate}")
        if strategy not in self.STRATEGIES:
            raise ValueError(f"strategy must be one of {self.STRATEGIES}, got {strategy!r}")

        self.rate = rate
        self.strategy = strategy
        self.noise_std = noise_std

        # Tracking
        self._poisoned_count: int = 0
        self._total_count: int = 0

    def poison(self, reward: float) -> float:
        """
        Optionally corrupt a scalar reward value.

        Parameters
        ----------
        reward : float
            Original reward from the environment.

        Returns
        -------
        float
            Possibly corrupted reward.
        """
        self._total_count += 1

        if random.random() >= self.rate:
            return reward          # not poisoned

        self._poisoned_count += 1

        if self.strategy == "flip":
            return -reward
        elif self.strategy == "zero":
            return 0.0
        else:  # "noise"
            return reward + random.gauss(0, self.noise_std)

    def poison_batch(self, rewards: np.ndarray) -> np.ndarray:
        """
        Corrupt a batch of rewards.

        Parameters
        ----------
        rewards : np.ndarray
            Shape (n,) reward array from a rollout buffer.

        Returns
        -------
        np.ndarray
            Poisoned reward array, same shape.
        """
        poisoned = rewards.copy()
        mask = np.random.random(len(rewards)) < self.rate

        if self.strategy == "flip":
            poisoned[mask] = -poisoned[mask]
        elif self.strategy == "zero":
            poisoned[mask] = 0.0
        else:  # "noise"
            poisoned[mask] += np.random.normal(0, self.noise_std, mask.sum())

        self._poisoned_count += int(mask.sum())
        self._total_count += len(rewards)

        return poisoned

    @property
    def poison_fraction(self) -> float:
        """Observed fraction of rewards that were poisoned so far."""
        if self._total_count == 0:
            return 0.0
        return self._poisoned_count / self._total_count

    def reset_stats(self) -> None:
        """Reset poisoning counters."""
        self._poisoned_count = 0
        self._total_count = 0

    def get_config(self) -> Dict[str, Any]:
        return {
            "type":             "RewardPoisoner",
            "rate":             self.rate,
            "strategy":         self.strategy,
            "noise_std":        self.noise_std,
            "poison_fraction":  round(self.poison_fraction, 4),
        }

    def __repr__(self) -> str:
        return (
            f"RewardPoisoner(rate={self.rate}, strategy={self.strategy!r}, "
            f"poisoned={self._poisoned_count}/{self._total_count})"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Observation Delay
# ══════════════════════════════════════════════════════════════════════════════

class ObservationDelayPerturbation(BasePerturbation):
    """
    Simulates stale / delayed observations.

    In a real network, telemetry data arrives with varying latency.
    An attacker who can delay IDS telemetry (e.g., by flooding the
    management network) causes the defender agent to act on outdated state.

    This perturbation maintains a circular buffer of past observations
    and returns a randomly chosen past observation instead of the current one.

    Parameters
    ----------
    max_delay : int
        Maximum number of steps of delay (buffer size).
    delay_prob : float
        Probability of returning a stale observation instead of current.
    """

    def __init__(
        self,
        max_delay: int = 5,
        delay_prob: float = 0.3,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
    ) -> None:
        super().__init__(clip_min, clip_max)
        if max_delay < 1:
            raise ValueError(f"max_delay must be >= 1, got {max_delay}")
        if not 0 <= delay_prob <= 1:
            raise ValueError(f"delay_prob must be in [0, 1], got {delay_prob}")

        self.max_delay = max_delay
        self.delay_prob = delay_prob

        # Circular observation history buffer
        self._buffer: list[torch.Tensor] = []

    def perturb(self, x: torch.Tensor, **_: Any) -> torch.Tensor:
        """
        Return either the current observation or a past one.

        Parameters
        ----------
        x : torch.Tensor
            Current observation tensor.

        Returns
        -------
        torch.Tensor
            Possibly stale observation of the same shape.
        """
        # Always push current obs into buffer
        self._buffer.append(x.detach().clone())
        if len(self._buffer) > self.max_delay:
            self._buffer.pop(0)

        # Randomly return a stale observation
        if len(self._buffer) > 1 and random.random() < self.delay_prob:
            # Pick a random past observation (not the most recent)
            idx = random.randint(0, len(self._buffer) - 2)
            return self._buffer[idx].clone()

        return x

    def reset_buffer(self) -> None:
        """Clear the observation history buffer (call at episode start)."""
        self._buffer.clear()

    def get_config(self) -> Dict[str, Any]:
        cfg = super().get_config()
        cfg.update({
            "max_delay":  self.max_delay,
            "delay_prob": self.delay_prob,
        })
        return cfg


# ══════════════════════════════════════════════════════════════════════════════
# Factory helper
# ══════════════════════════════════════════════════════════════════════════════

def build_perturbation(config: Dict[str, Any]) -> BasePerturbation:
    """
    Build a perturbation instance from a config dict.

    Parameters
    ----------
    config : dict
        Must contain a ``"type"`` key matching one of the perturbation
        class names.  Other keys are passed as constructor kwargs.

    Returns
    -------
    BasePerturbation

    Example
    -------
        cfg = {"type": "PGDPerturbation", "epsilon": 0.1, "steps": 10}
        perturb = build_perturbation(cfg)
    """
    registry = {
        "FGSMPerturbation":              FGSMPerturbation,
        "PGDPerturbation":               PGDPerturbation,
        "GaussianNoisePerturbation":     GaussianNoisePerturbation,
        "ObservationDelayPerturbation":  ObservationDelayPerturbation,
    }

    ptype = config.pop("type", None)
    if ptype not in registry:
        raise ValueError(
            f"Unknown perturbation type {ptype!r}. "
            f"Available: {list(registry.keys())}"
        )

    return registry[ptype](**config)