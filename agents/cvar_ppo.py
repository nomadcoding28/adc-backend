"""
CVaR-PPO: Risk-Sensitive Proximal Policy Optimization
======================================================
Core Novelty 3 of the ACD Framework.

This module replaces the standard PPO value loss with a
Conditional Value-at-Risk (CVaR) weighted objective, making
the agent explicitly optimize for worst-case cyber scenarios
rather than average performance.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THE MATH
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Standard PPO loss:
    L = L_PG  +  c₁·L_VF  -  c₂·H(π)

    where L_VF = E[(V_θ(sₜ) - Rₜ)²]   (mean squared TD error)

Problem: minimizing MEAN squared error means the agent trades
off catastrophic failures for average gains — fine for games,
unacceptable for cyber defence where a single breach is fatal.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CVaR-PPO loss:
    L = L_PG_CVaR  +  c₁·L_VF_CVaR  -  c₂·H(π)  +  λ·L_EWC

Step 1 — Compute VaR (Value-at-Risk):
    VaR_α = α-quantile of returns in the current minibatch
    (e.g. α=0.05 → 5th percentile of returns)

Step 2 — Identify tail samples:
    Tα = { i : Rᵢ ≤ VaR_α }   (worst α-fraction of the batch)

Step 3 — Compute CVaR importance weights:
            ⎧ 1/α   if Rᵢ ≤ VaR_α   (in the tail)
    wᵢ =  ⎨
            ⎩ 0     otherwise         (outside the tail)

Step 4 — CVaR value loss (critic focuses on tail):
    L_VF_CVaR = E[wᵢ · (V_θ(sᵢ) - Rᵢ)²]

Step 5 — CVaR policy gradient (actor also tail-weighted):
    L_PG_CVaR = E[wᵢ · min(rᵢ·Aᵢ, clip(rᵢ, 1-ε, 1+ε)·Aᵢ)]

    where rᵢ = π_θ(aᵢ|sᵢ) / π_θ_old(aᵢ|sᵢ)  (probability ratio)
          Aᵢ = GAE advantage estimate

Step 6 — EWC penalty (catastrophic forgetting prevention):
    L_EWC = (λ/2) · Σᵢ Fᵢ·(θᵢ - θ*ᵢ)²

Full combined objective:
    L_total = L_PG_CVaR + c₁·L_VF_CVaR - c₂·H(π) + λ·L_EWC

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHY THIS WORKS IN CYBER DEFENCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Standard RL optimises:   max E[R]     ← average performance
CVaR-RL optimises:       max CVaR_α[R] ← worst-case performance

In cyber defence:
- "Average case" might be 95% success with 5% catastrophic breach
- Standard PPO is happy with this — the average reward is high
- CVaR-PPO focuses on that 5% and tries to eliminate the breach

α = 0.05 means: "I care ONLY about the worst 5% of outcomes."
Lower α = more risk-averse = more conservative defender.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

IMPLEMENTATION NOTE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SB3's PPO.train() is a single method that computes all three
loss terms in one loop. To inject CVaR without forking SB3,
we subclass PPO and override train() precisely, copying the
original logic and substituting only the loss computation.

References:
- Rockafellar & Uryasev (2000) — CVaR definition & properties
- Tamar et al. (2015) — Policy gradient for CVaR
- Chow & Ghavamzadeh (2014) — CVaR actor-critic
"""

import os
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from typing import Optional, Dict, Tuple, List, Union, Type
from collections import deque

from stable_baselines3 import PPO
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback
from stable_baselines3.common.utils import explained_variance, get_schedule_fn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from loguru import logger


# ══════════════════════════════════════════════════════════════════════
#  Feature Extractor & Policy  (unchanged from scaffold)
# ══════════════════════════════════════════════════════════════════════

class ACDFeatureExtractor(BaseFeaturesExtractor):
    """
    Deep feature extractor for ACD observations.

    Input: CybORG observation (52-dim) + KG enrichment (16-dim) = 68-dim
    Output: 256-dim feature vector fed to both actor and critic heads.

    Architecture chosen to capture:
    - Low-level: host compromise indicators
    - Mid-level: lateral movement patterns
    - High-level: campaign-level threat context (from KG)
    """

    def __init__(self, observation_space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        obs_dim = observation_space.shape[0]

        self.network = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU(),
        )

        # Weight initialisation — orthogonal as recommended for PPO
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(observations)


class ACDActorCriticPolicy(ActorCriticPolicy):
    """ACD Actor-Critic with separate deep networks for π and V."""

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            features_extractor_class=ACDFeatureExtractor,
            features_extractor_kwargs={"features_dim": 256},
            # Separate networks for actor and critic
            net_arch=dict(pi=[256, 128], vf=[256, 128]),
            activation_fn=nn.ReLU,
        )


# ══════════════════════════════════════════════════════════════════════
#  CVaR Computation Utilities
# ══════════════════════════════════════════════════════════════════════

class CVaRComputer:
    """
    Stateless utility class for CVaR computations.
    All methods operate on torch tensors for GPU compatibility.
    """

    @staticmethod
    def compute_var(returns: torch.Tensor, alpha: float) -> torch.Tensor:
        """
        Compute Value-at-Risk (α-quantile) of the return distribution.

        VaR_α(R) = inf{ r : P(R ≤ r) ≥ α }

        Args:
            returns: Tensor of shape (N,) — batch of return values
            alpha:   Risk level in (0, 1)

        Returns:
            Scalar tensor — the α-quantile
        """
        return torch.quantile(returns, alpha)

    @staticmethod
    def compute_cvar(returns: torch.Tensor, alpha: float) -> torch.Tensor:
        """
        Compute Conditional Value-at-Risk (expected shortfall).

        CVaR_α(R) = E[R | R ≤ VaR_α(R)]
                  = (1/α) · E[R · 1{R ≤ VaR_α}]

        This is the expected return in the worst α-fraction.

        Args:
            returns: Tensor of shape (N,)
            alpha:   Risk level in (0, 1)

        Returns:
            Scalar tensor — the CVaR value
        """
        var = CVaRComputer.compute_var(returns, alpha)
        # Boolean mask for tail samples
        tail_mask = (returns <= var).float()
        n_tail = tail_mask.sum().clamp(min=1.0)
        # Mean of tail returns
        cvar = (returns * tail_mask).sum() / n_tail
        return cvar

    @staticmethod
    def compute_importance_weights(
        returns: torch.Tensor,
        alpha: float,
        normalise: bool = True,
    ) -> torch.Tensor:
        """
        Compute CVaR importance weights for each sample.

        The exact CVaR importance weighting:
                ⎧ 1/α    if Rᵢ ≤ VaR_α    ← tail samples (amplified)
        wᵢ  =  ⎨
                ⎩ 0      otherwise          ← non-tail samples (ignored)

        After normalisation (so weights sum to batch_size):
                ⎧ N / (α · N_tail)   if in tail
        wᵢ  =  ⎨
                ⎩ 0                  otherwise

        This ensures:
        - Tail samples contribute fully to the gradient
        - Non-tail samples are excluded from the value update
        - The total weight magnitude stays comparable to standard loss

        Args:
            returns:   Tensor of shape (N,)
            alpha:     Risk level
            normalise: Whether to normalise weights to mean=1

        Returns:
            weights: Tensor of shape (N,) — non-negative floats
        """
        var = CVaRComputer.compute_var(returns, alpha)
        tail_mask = (returns <= var).float()
        n_tail = tail_mask.sum().clamp(min=1.0)

        # Raw weights: 1/α for tail, 0 otherwise
        weights = tail_mask / alpha

        if normalise:
            # Rescale so mean weight ≈ 1.0 for stable training
            N = float(returns.shape[0])
            weights = weights * (N / n_tail)

        return weights

    @staticmethod
    def compute_soft_weights(
        returns: torch.Tensor,
        alpha: float,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Soft CVaR weights — differentiable approximation using sigmoid.

        Useful when you want gradients to flow through the weight computation.
        Standard hard weights are non-differentiable at VaR_α.

        wᵢ = σ((VaR_α - Rᵢ) / temperature) / α

        Args:
            returns:     Tensor of shape (N,)
            alpha:       Risk level
            temperature: Sharpness of transition (lower → harder cutoff)

        Returns:
            weights: Tensor of shape (N,)
        """
        var = CVaRComputer.compute_var(returns, alpha)
        soft_mask = torch.sigmoid((var - returns) / temperature)
        weights = soft_mask / alpha
        # Normalise
        weights = weights / weights.mean().clamp(min=1e-8)
        return weights


# ══════════════════════════════════════════════════════════════════════
#  CVaR-PPO  (the actual research contribution)
# ══════════════════════════════════════════════════════════════════════

class CVaRPPO(PPO):
    """
    CVaR-PPO: Risk-Sensitive Proximal Policy Optimization.

    Subclasses SB3's PPO and overrides the train() method to
    replace standard value loss and policy gradient with
    CVaR-weighted equivalents.

    Key differences from standard PPO:
    ┌────────────────────────┬──────────────────┬──────────────────────┐
    │ Component              │ Standard PPO     │ CVaR-PPO             │
    ├────────────────────────┼──────────────────┼──────────────────────┤
    │ Value loss             │ MSE(V(s), R)     │ CVaR-weighted MSE    │
    │ Policy gradient        │ E[clip·A]        │ E[w·clip·A]          │
    │ Optimisation target    │ E[R]             │ CVaR_α[R]            │
    │ Failure handling       │ Averages out     │ Explicitly penalised │
    │ EWC regularisation     │ ✗                │ ✓                    │
    └────────────────────────┴──────────────────┴──────────────────────┘

    Usage:
        model = CVaRPPO(
            policy=ACDActorCriticPolicy,
            env=env,
            cvar_alpha=0.05,
            cvar_weight=0.3,
            use_cvar_policy_gradient=True,
        )
        model.learn(total_timesteps=1_000_000)
    """

    def __init__(
        self,
        policy,
        env: GymEnv,
        # ── CVaR-specific parameters ──────────────────────────────
        cvar_alpha: float = 0.05,
        cvar_weight: float = 0.3,
        use_cvar_policy_gradient: bool = True,
        soft_weights: bool = False,
        weight_temperature: float = 0.5,
        # ── EWC parameters ────────────────────────────────────────
        ewc_module=None,
        # ── Standard PPO parameters ───────────────────────────────
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range=0.2,
        clip_range_vf=None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        target_kl: Optional[float] = None,
        tensorboard_log: Optional[str] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[torch.device, str] = "auto",
        **kwargs,
    ):
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            normalize_advantage=normalize_advantage,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            target_kl=target_kl,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            seed=seed,
            device=device,
            **kwargs,
        )

        # CVaR parameters
        self.cvar_alpha = cvar_alpha
        self.cvar_weight = cvar_weight          # Weight of CVaR vs standard loss
        self.use_cvar_pg = use_cvar_policy_gradient
        self.soft_weights = soft_weights
        self.weight_temperature = weight_temperature

        # EWC module (injected externally, optional)
        self.ewc_module = ewc_module

        # Metrics tracking
        self._cvar_metrics = CVaRMetricsTracker(window_size=1000)

        logger.info(
            f"CVaRPPO initialised | α={cvar_alpha} | "
            f"cvar_weight={cvar_weight} | "
            f"cvar_pg={use_cvar_policy_gradient} | "
            f"ewc={'enabled' if ewc_module else 'disabled'}"
        )

    def attach_ewc(self, ewc_module):
        """Attach EWC module for continual learning regularisation."""
        self.ewc_module = ewc_module
        logger.info("EWC module attached to CVaRPPO")

    def train(self) -> None:
        """
        CVaR-PPO training update.

        Overrides PPO.train() to inject CVaR-weighted loss.
        Called automatically by SB3 after each rollout collection.

        The structure mirrors SB3's original train() exactly,
        with three surgical modifications:
          1. Compute CVaR weights from the current minibatch returns
          2. Apply weights to value loss
          3. Apply weights to policy gradient (if enabled)
          4. Add EWC penalty (if enabled)
        """
        # ── Setup (identical to SB3 PPO.train()) ──────────────────
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses, pg_losses, value_losses = [], [], []
        cvar_losses, ewc_losses, total_losses = [], [], []
        clip_fractions = []
        continue_training = True

        # ── Training epochs ────────────────────────────────────────
        for epoch in range(self.n_epochs):
            approx_kl_divs = []

            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space,
                              __import__('gymnasium').spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                # ── Forward pass ───────────────────────────────────
                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions
                )
                values = values.flatten()

                # ── Advantages ─────────────────────────────────────
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-8
                    )

                # ── Returns (used for CVaR weights) ────────────────
                returns = rollout_data.returns.flatten()

                # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                # STEP 1: Compute CVaR importance weights
                # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                with torch.no_grad():
                    if self.soft_weights:
                        cvar_weights = CVaRComputer.compute_soft_weights(
                            returns,
                            self.cvar_alpha,
                            self.weight_temperature,
                        )
                    else:
                        cvar_weights = CVaRComputer.compute_importance_weights(
                            returns,
                            self.cvar_alpha,
                            normalise=True,
                        )

                    # Compute current CVaR for logging
                    current_cvar = CVaRComputer.compute_cvar(
                        returns, self.cvar_alpha
                    )

                # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                # STEP 2: Policy gradient loss (clipped surrogate)
                # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                ratio = torch.exp(log_prob - rollout_data.old_log_prob)

                # Clipped surrogate objective
                pg_loss_unweighted_1 = advantages * ratio
                pg_loss_unweighted_2 = advantages * torch.clamp(
                    ratio, 1 - clip_range, 1 + clip_range
                )
                pg_loss_per_sample = -torch.min(
                    pg_loss_unweighted_1, pg_loss_unweighted_2
                )

                if self.use_cvar_pg:
                    # ─── CVaR-weighted policy gradient ──────────────
                    # Weight the surrogate loss by CVaR importance.
                    # This forces the actor to improve on tail episodes
                    # more aggressively than average episodes.
                    pg_loss = (cvar_weights * pg_loss_per_sample).mean()
                else:
                    pg_loss = pg_loss_per_sample.mean()

                # Clip fraction (for logging)
                with torch.no_grad():
                    clip_fraction = torch.mean(
                        (torch.abs(ratio - 1) > clip_range).float()
                    ).item()
                    clip_fractions.append(clip_fraction)

                # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                # STEP 3: Value loss — CVaR-weighted MSE
                # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                if self.clip_range_vf is None:
                    # ─── Standard value prediction error ────────────
                    values_pred = values
                else:
                    # ─── Clipped value prediction ────────────────────
                    values_pred = rollout_data.old_values + torch.clamp(
                        values - rollout_data.old_values,
                        -clip_range_vf, clip_range_vf,
                    )

                # Per-sample squared TD error
                td_error_sq = F.mse_loss(values_pred, returns, reduction="none")

                # ─── Standard MSE value loss ──────────────────────────
                value_loss_standard = td_error_sq.mean()

                # ─── CVaR-weighted value loss ─────────────────────────
                # Amplifies the critic's error on tail (worst-case) states.
                # The critic learns to be accurate where it matters most:
                # the catastrophic failure states.
                value_loss_cvar = (cvar_weights * td_error_sq).mean()

                # ─── Interpolated value loss ──────────────────────────
                # cvar_weight controls the trade-off:
                #   0.0 → pure standard MSE  (risk-neutral)
                #   1.0 → pure CVaR loss     (fully risk-averse)
                value_loss = (
                    (1.0 - self.cvar_weight) * value_loss_standard
                    + self.cvar_weight * value_loss_cvar
                )

                # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                # STEP 4: Entropy bonus (unchanged from standard PPO)
                # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                if entropy is None:
                    entropy_loss = -log_prob.mean()
                else:
                    entropy_loss = -entropy.mean()

                # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                # STEP 5: EWC penalty (Novelty 1 integration)
                # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                ewc_loss = torch.tensor(0.0, device=self.device)
                if self.ewc_module is not None:
                    ewc_loss = self.ewc_module.penalty(self.policy)

                # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                # STEP 6: Combined loss
                # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                #
                #  L = L_PG_CVaR
                #    + vf_coef · L_VF_CVaR
                #    - ent_coef · H(π)
                #    + L_EWC
                #
                loss = (
                    pg_loss
                    + self.vf_coef * value_loss
                    - self.ent_coef * entropy_loss
                    + ewc_loss
                )

                # ── KL divergence check (early stopping) ───────────
                with torch.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = (
                        (torch.exp(log_ratio) - 1) - log_ratio
                    ).mean().item()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        logger.info(
                            f"Early stopping at epoch {epoch} "
                            f"due to KL divergence: {approx_kl_div:.4f}"
                        )
                    break

                # ── Backward pass ──────────────────────────────────
                self.policy.optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

                # ── Track metrics ──────────────────────────────────
                entropy_losses.append(entropy_loss.item())
                pg_losses.append(pg_loss.item())
                value_losses.append(value_loss.item())
                cvar_losses.append(value_loss_cvar.item())
                ewc_losses.append(ewc_loss.item())
                total_losses.append(loss.item())

                # Track CVaR metrics
                self._cvar_metrics.update(
                    returns=returns.detach().cpu().numpy(),
                    cvar_value=current_cvar.item(),
                    weights=cvar_weights.detach().cpu().numpy(),
                )

            if not continue_training:
                break

        self._n_updates += self.n_epochs
        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(),
            self.rollout_buffer.returns.flatten(),
        )

        # ── Log all metrics ────────────────────────────────────────
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/cvar_value_loss", np.mean(cvar_losses))
        self.logger.record("train/ewc_loss", np.mean(ewc_losses))
        self.logger.record("train/total_loss", np.mean(total_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", np.mean(total_losses))
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)

        # ── CVaR-specific logs ─────────────────────────────────────
        cvar_summary = self._cvar_metrics.get_summary()
        self.logger.record("cvar/running_cvar", cvar_summary["mean_cvar"])
        self.logger.record("cvar/tail_sample_fraction", cvar_summary["tail_fraction"])
        self.logger.record("cvar/mean_weight_tail", cvar_summary["mean_weight_tail"])
        self.logger.record("cvar/alpha", self.cvar_alpha)

    def get_cvar_metrics(self) -> Dict:
        """Return current CVaR training metrics."""
        return self._cvar_metrics.get_summary()


# ══════════════════════════════════════════════════════════════════════
#  Metrics Tracker
# ══════════════════════════════════════════════════════════════════════

class CVaRMetricsTracker:
    """
    Tracks CVaR-related training metrics over a rolling window.

    Provides:
    - Running CVaR estimate
    - Fraction of tail samples per batch
    - Mean weight of tail vs non-tail samples
    - Historical CVaR curve (for plotting)
    """

    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self._cvar_history: deque = deque(maxlen=window_size)
        self._tail_fractions: deque = deque(maxlen=window_size)
        self._mean_weights_tail: deque = deque(maxlen=window_size)
        self._returns_history: deque = deque(maxlen=window_size * 64)

    def update(
        self,
        returns: np.ndarray,
        cvar_value: float,
        weights: np.ndarray,
    ):
        self._cvar_history.append(cvar_value)

        # Fraction of samples that are in the tail (weight > 0)
        tail_fraction = float((weights > 0).mean())
        self._tail_fractions.append(tail_fraction)

        # Mean weight of non-zero (tail) samples
        tail_weights = weights[weights > 0]
        if len(tail_weights) > 0:
            self._mean_weights_tail.append(float(tail_weights.mean()))

        for r in returns:
            self._returns_history.append(float(r))

    def get_summary(self) -> Dict:
        if not self._cvar_history:
            return {
                "mean_cvar": 0.0,
                "tail_fraction": 0.0,
                "mean_weight_tail": 0.0,
                "n_updates": 0,
            }
        return {
            "mean_cvar": float(np.mean(list(self._cvar_history))),
            "latest_cvar": float(self._cvar_history[-1]),
            "tail_fraction": float(np.mean(list(self._tail_fractions))),
            "mean_weight_tail": float(
                np.mean(list(self._mean_weights_tail))
                if self._mean_weights_tail else 1.0
            ),
            "n_updates": len(self._cvar_history),
        }

    def get_cvar_curve(self) -> List[float]:
        """Returns the full CVaR history for plotting."""
        return list(self._cvar_history)


# ══════════════════════════════════════════════════════════════════════
#  ACDPPOAgent  (updated to use CVaRPPO)
# ══════════════════════════════════════════════════════════════════════

class ACDPPOAgent:
    """
    ACD Agent wrapper — now uses CVaRPPO as the backend.

    Provides a clean interface to CVaRPPO with:
    - Config-driven construction
    - EWC injection
    - Checkpoint management
    - Training orchestration
    """

    def __init__(self, env: GymEnv, config: Dict):
        self.env = env
        self.config = config
        ppo_cfg = config.get("ppo", {})
        cvar_cfg = config.get("cvar", {})

        cvar_enabled = cvar_cfg.get("enabled", True)

        if cvar_enabled:
            self.model = CVaRPPO(
                policy=ACDActorCriticPolicy,
                env=env,
                # CVaR params
                cvar_alpha=cvar_cfg.get("alpha", 0.05),
                cvar_weight=cvar_cfg.get("weight", 0.3),
                use_cvar_policy_gradient=cvar_cfg.get(
                    "use_cvar_policy_gradient", True
                ),
                soft_weights=cvar_cfg.get("soft_weights", False),
                # PPO params
                learning_rate=ppo_cfg.get("learning_rate", 3e-4),
                n_steps=ppo_cfg.get("n_steps", 2048),
                batch_size=ppo_cfg.get("batch_size", 64),
                n_epochs=ppo_cfg.get("n_epochs", 10),
                gamma=ppo_cfg.get("gamma", 0.99),
                gae_lambda=ppo_cfg.get("gae_lambda", 0.95),
                clip_range=ppo_cfg.get("clip_range", 0.2),
                ent_coef=ppo_cfg.get("ent_coef", 0.01),
                vf_coef=ppo_cfg.get("vf_coef", 0.5),
                max_grad_norm=ppo_cfg.get("max_grad_norm", 0.5),
                verbose=ppo_cfg.get("verbose", 1),
                tensorboard_log=config.get("logging", {}).get(
                    "tensorboard_dir", "./logs/tensorboard"
                ),
                device=ppo_cfg.get("device", "auto"),
            )
            logger.info(
                f"ACDPPOAgent using CVaRPPO | "
                f"α={cvar_cfg.get('alpha', 0.05)}"
            )
        else:
            # Fallback to standard PPO for ablation studies
            self.model = PPO(
                policy=ACDActorCriticPolicy,
                env=env,
                learning_rate=ppo_cfg.get("learning_rate", 3e-4),
                n_steps=ppo_cfg.get("n_steps", 2048),
                batch_size=ppo_cfg.get("batch_size", 64),
                n_epochs=ppo_cfg.get("n_epochs", 10),
                gamma=ppo_cfg.get("gamma", 0.99),
                ent_coef=ppo_cfg.get("ent_coef", 0.01),
                vf_coef=ppo_cfg.get("vf_coef", 0.5),
                verbose=1,
                device=ppo_cfg.get("device", "auto"),
            )
            logger.info("ACDPPOAgent using standard PPO (CVaR disabled)")

    def attach_ewc(self, ewc_module):
        """Wire EWC into the PPO training loop."""
        if isinstance(self.model, CVaRPPO):
            self.model.attach_ewc(ewc_module)
        else:
            logger.warning("EWC requires CVaRPPO — ignored for standard PPO")

    # Keep old name for backward compatibility
    def set_ewc(self, ewc_module):
        self.attach_ewc(ewc_module)

    def train(
        self,
        total_timesteps: int,
        callback=None,
        reset_num_timesteps: bool = True,
    ):
        logger.info(f"Training | timesteps={total_timesteps:,}")
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            reset_num_timesteps=reset_num_timesteps,
            tb_log_name="ACD_CVaR_PPO",
        )

    def predict(self, observation: np.ndarray, deterministic: bool = False):
        return self.model.predict(observation, deterministic=deterministic)

    def save(self, path: str):
        self.model.save(path)
        logger.info(f"Checkpoint saved → {path}")

    def load(self, path: str):
        if isinstance(self.model, CVaRPPO):
            self.model = CVaRPPO.load(path, env=self.env)
        else:
            self.model = PPO.load(path, env=self.env)
        logger.info(f"Checkpoint loaded ← {path}")

    def get_cvar_metrics(self) -> Dict:
        if isinstance(self.model, CVaRPPO):
            return self.model.get_cvar_metrics()
        return {}

    def get_value_estimate(self, obs: np.ndarray) -> float:
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.model.device)
        with torch.no_grad():
            _, value, _ = self.model.policy.forward(obs_t)
        return value.item()

    @property
    def policy(self):
        return self.model.policy

    @property
    def num_timesteps(self) -> int:
        return self.model.num_timesteps

    @property
    def is_cvar(self) -> bool:
        return isinstance(self.model, CVaRPPO)

    # ── Lifecycle methods (expected by api/routers/training.py) ────────

    def start_training(self) -> None:
        """Mark the agent as currently training."""
        self._is_training = True

    def stop_training(self) -> None:
        """Mark the agent as not training."""
        self._is_training = False

    @property
    def is_training(self) -> bool:
        return getattr(self, "_is_training", False)

    @is_training.setter
    def is_training(self, value: bool):
        self._is_training = value

    @property
    def total_timesteps_trained(self) -> int:
        return self.model.num_timesteps

    def learn(self, total_timesteps: int, **kwargs):
        """Delegate to self.train() for API compatibility."""
        self.train(total_timesteps=total_timesteps, **kwargs)

    def get_metrics(self) -> Dict:
        """Return current training metrics snapshot."""
        metrics = {
            "agent_type": "cvar_ppo" if self.is_cvar else "ppo",
            "device": str(self.model.device),
            "total_timesteps": self.model.num_timesteps,
            "is_training": self.is_training,
        }
        if self.is_cvar:
            metrics.update(self.get_cvar_metrics())
        return metrics

    def save_checkpoint(self, tag: str = "") -> str:
        """Save model checkpoint and return the path."""
        checkpoint_dir = self.config.get("checkpoint_dir", "data/checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        suffix = f"_{tag}" if tag else ""
        path = os.path.join(checkpoint_dir, f"cvar_ppo{suffix}")
        self.save(path)
        return path