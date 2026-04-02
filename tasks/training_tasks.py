"""
tasks/training_tasks.py
========================
Celery tasks for RL agent training — CVaR-PPO + EWC.

Tasks
-----
    run_training(config, total_timesteps)
        Full training run.  Streams progress every 10k steps via
        Celery task state so GET /training/status shows live metrics.
        Catches SoftTimeLimitExceeded (2h) and saves a checkpoint.

    run_adversarial_training(config, total_timesteps)
        Min-max adversarial training (Novelty 2).
        Applies FGSM / PGD / noise / reward-poisoning perturbations.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

from celery import Task
from celery.exceptions import SoftTimeLimitExceeded

from tasks.celery_app import celery_app

logger = logging.getLogger(__name__)

_UPDATE_INTERVAL = 10_000   # steps between Celery state updates


class _BaseTrainingTask(Task):
    """
    Base task that cleans up agent + env on any exit (success/failure/revoke).
    """
    abstract = True

    def __init__(self) -> None:
        super().__init__()
        self._agent = None
        self._env   = None

    def after_return(self, status, retval, task_id, args, kwargs, einfo) -> None:
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
            self._env = None
        if self._agent is not None:
            try:
                if hasattr(self._agent, "stop_training"):
                    self._agent.stop_training()
            except Exception:
                pass
            self._agent = None


@celery_app.task(
    bind          = True,
    base          = _BaseTrainingTask,
    name          = "tasks.training_tasks.run_training",
    queue         = "training",
    max_retries   = 0,
    track_started = True,
    ignore_result = False,
)
def run_training(
    self:            Task,
    config:          Dict[str, Any],
    total_timesteps: int = 2_000_000,
) -> Dict[str, Any]:
    """
    Execute a full CVaR-PPO + EWC training run.

    Parameters
    ----------
    config : dict
        Full training configuration (agent section from config.yaml plus
        any overrides from the API request body).
    total_timesteps : int
        Total CybORG environment interaction steps.  Default 2 million.

    Returns
    -------
    dict
        Keys: status, run_id, total_timesteps, mean_reward, cvar_005,
        cvar_001, catastrophic_rate, ewc_forgetting, ewc_tasks,
        drift_events, elapsed_s, checkpoint_path.
    """
    run_id     = (self.request.id or "local")[:8]
    start_time = time.monotonic()
    ckpt_path  = None

    logger.info(
        "Training task START — run_id=%s  steps=%d  agent=%s",
        run_id, total_timesteps, config.get("agent_type", "cvar_ppo"),
    )

    self.update_state(
        state = "STARTED",
        meta  = {
            "run_id":  run_id,
            "step":    0,
            "total":   total_timesteps,
            "pct":     0.0,
            "message": "Initialising agent and environment...",
        },
    )

    try:
        import numpy as np
        from envs.env_factory import make_env
        from agents.registry import AgentRegistry
        from drift.detector_factory import DetectorFactory
        from monitoring.prometheus import PrometheusMetrics

        prom        = PrometheusMetrics()
        self._env   = make_env(config, n_envs=config.get("n_envs", 1))
        self._agent = AgentRegistry.build(self._env, config)
        drift_det   = DetectorFactory.build(config.get("drift", {}))

        # Resume from checkpoint if configured
        resume = config.get("resume_checkpoint")
        if resume:
            from pathlib import Path
            if Path(resume).exists():
                self._agent.model = self._agent.model.__class__.load(
                    resume, env=self._env
                )
                logger.info("Resumed from: %s", resume)

        self._agent.start_training()
        prom.set_training_active(True)

        # ── SB3 progress callback ─────────────────────────────────────
        last_update = [0]

        def _progress_callback(locals_: dict, globals_: dict) -> Optional[bool]:
            model = locals_.get("self")
            if model is None:
                return None
            step = getattr(model, "num_timesteps", 0)
            if step - last_update[0] < _UPDATE_INTERVAL:
                return None
            last_update[0] = step

            ep_buf   = getattr(model, "ep_info_buffer", [])
            mean_r   = None
            cvar_val = None
            if ep_buf:
                rewards  = [info["r"] for info in ep_buf]
                mean_r   = float(np.mean(rewards))
                n_tail   = max(1, int(0.05 * len(rewards)))
                cvar_val = float(np.mean(np.sort(rewards)[:n_tail]))

            pct = round(step / max(total_timesteps, 1) * 100, 1)

            self.update_state(
                state = "PROGRESS",
                meta  = {
                    "run_id":      run_id,
                    "step":        step,
                    "total":       total_timesteps,
                    "pct":         pct,
                    "mean_reward": mean_r,
                    "cvar_005":    cvar_val,
                    "episode":     len(ep_buf),
                },
            )

            if mean_r   is not None: prom.update_reward(mean_r)
            if cvar_val is not None: prom.update_cvar(0.05, cvar_val)
            prom.increment_timesteps(_UPDATE_INTERVAL)
            return None

        # ── Train ─────────────────────────────────────────────────────
        try:
            self._agent.learn(
                total_timesteps = total_timesteps,
                progress_bar    = False,
                callback        = _progress_callback,
            )
        except SoftTimeLimitExceeded:
            logger.warning("Soft time limit reached — saving checkpoint.")
            ckpt_path = str(self._agent.save_checkpoint(tag="time_limit"))
            prom.set_training_active(False)
            return {
                "status":          "time_limit",
                "run_id":          run_id,
                "total_timesteps": self._agent.total_timesteps_trained,
                "checkpoint_path": ckpt_path,
                "elapsed_s":       round(time.monotonic() - start_time, 1),
                "message":         "Stopped at 2-hour limit. Checkpoint saved.",
            }

        # ── Save checkpoint + collect metrics ──────────────────────────
        ckpt_path     = str(self._agent.save_checkpoint(tag="final"))
        final_metrics = self._agent.get_metrics()
        elapsed       = round(time.monotonic() - start_time, 1)

        prom.update_from_agent_metrics(final_metrics)
        prom.set_training_active(False)

        logger.info(
            "Training COMPLETE — run_id=%s  steps=%d  mean_reward=%.4f  "
            "cvar_005=%.4f  elapsed=%.1fs",
            run_id,
            final_metrics.get("total_timesteps", 0),
            final_metrics.get("mean_reward") or 0.0,
            final_metrics.get("cvar_005")    or 0.0,
            elapsed,
        )

        return {
            "status":            "completed",
            "run_id":            run_id,
            "agent_type":        config.get("agent_type", "cvar_ppo"),
            "total_timesteps":   final_metrics.get("total_timesteps", 0),
            "mean_reward":       final_metrics.get("mean_reward"),
            "cvar_005":          final_metrics.get("cvar_005"),
            "cvar_001":          final_metrics.get("cvar_001"),
            "catastrophic_rate": final_metrics.get("catastrophic_rate"),
            "ewc_forgetting":    final_metrics.get("ewc_forgetting"),
            "ewc_tasks":         final_metrics.get("ewc_tasks_registered", 0),
            "drift_events":      drift_det.n_events if drift_det else 0,
            "elapsed_s":         elapsed,
            "checkpoint_path":   ckpt_path,
        }

    except Exception as exc:
        elapsed = round(time.monotonic() - start_time, 1)
        logger.error("Training FAILED — %s  elapsed=%.1fs", exc, elapsed, exc_info=True)
        if self._agent is not None:
            try:
                ckpt_path = str(self._agent.save_checkpoint(tag="error"))
                logger.info("Emergency checkpoint: %s", ckpt_path)
            except Exception:
                pass
        raise


@celery_app.task(
    bind          = True,
    base          = _BaseTrainingTask,
    name          = "tasks.training_tasks.run_adversarial_training",
    queue         = "training",
    max_retries   = 0,
    track_started = True,
)
def run_adversarial_training(
    self:            Task,
    config:          Dict[str, Any],
    total_timesteps: int = 2_000_000,
) -> Dict[str, Any]:
    """
    Adversarial min-max training (Novelty 2).

    Applies FGSM / PGD / Gaussian noise observation perturbations and
    optional reward poisoning at adversarial_ratio of training steps
    to build robustness against observation corruption and reward hacking.

    Parameters
    ----------
    config : dict
        Training config.  Key sub-section: ``config["adversarial"]``.
    total_timesteps : int

    Returns
    -------
    dict
        Training result dict with checkpoint_path.
    """
    run_id     = (self.request.id or "local")[:8]
    start_time = time.monotonic()
    adv_cfg    = config.get("adversarial", {})

    logger.info(
        "Adversarial training START — run_id=%s  steps=%d  ratio=%.0f%%",
        run_id, total_timesteps, adv_cfg.get("adversarial_ratio", 0.3) * 100,
    )

    self.update_state(
        state = "STARTED",
        meta  = {
            "run_id":  run_id,
            "mode":    "adversarial",
            "step":    0,
            "total":   total_timesteps,
            "message": "Initialising adversarial trainer...",
        },
    )

    try:
        from envs.env_factory import make_env
        from agents.registry import AgentRegistry
        from agents.adversarial_trainer import AdversarialTrainer
        from agents.perturbation import (
            FGSMPerturbation,
            PGDPerturbation,
            GaussianNoisePerturbation,
            RewardPoisoningPerturbation,
        )
        from monitoring.prometheus import PrometheusMetrics

        self._env   = make_env(config, n_envs=1)
        self._agent = AgentRegistry.build(self._env, config)

        perturbations = [
            FGSMPerturbation(epsilon=adv_cfg.get("fgsm_epsilon", 0.10)),
            PGDPerturbation(
                epsilon = adv_cfg.get("pgd_epsilon", 0.10),
                steps   = adv_cfg.get("pgd_steps",   10),
            ),
            GaussianNoisePerturbation(std=adv_cfg.get("noise_std", 0.05)),
        ]

        poison_cfg = adv_cfg.get("reward_poison", {})
        if poison_cfg.get("rate", 0) > 0:
            perturbations.append(
                RewardPoisoningPerturbation(
                    rate     = poison_cfg.get("rate",     0.05),
                    strategy = poison_cfg.get("strategy", "flip"),
                )
            )

        trainer = AdversarialTrainer(
            agent             = self._agent,
            obs_perturbations = perturbations,
            adversarial_ratio = adv_cfg.get("adversarial_ratio", 0.3),
            config            = config,
        )

        try:
            result = trainer.train(total_timesteps=total_timesteps)
        except SoftTimeLimitExceeded:
            ckpt = str(self._agent.save_checkpoint(tag="adv_time_limit"))
            logger.warning("Adversarial training time limit — ckpt: %s", ckpt)
            return {
                "status":          "time_limit",
                "run_id":          run_id,
                "checkpoint_path": ckpt,
                "elapsed_s":       round(time.monotonic() - start_time, 1),
            }

        ckpt    = str(self._agent.save_checkpoint(tag="adversarial_final"))
        elapsed = round(time.monotonic() - start_time, 1)

        PrometheusMetrics().update_from_agent_metrics(self._agent.get_metrics())

        logger.info("Adversarial training COMPLETE — run_id=%s  elapsed=%.1fs", run_id, elapsed)

        return {
            "status":          "completed",
            "run_id":          run_id,
            "mode":            "adversarial",
            "elapsed_s":       elapsed,
            "checkpoint_path": ckpt,
            **result,
        }

    except Exception as exc:
        logger.error("Adversarial training FAILED: %s", exc, exc_info=True)
        raise