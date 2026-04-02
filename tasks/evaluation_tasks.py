"""
tasks/evaluation_tasks.py
==========================
Celery tasks for asynchronous evaluation and benchmarking.

Tasks
-----
    run_evaluation(config, n_episodes, checkpoint_path)
        Run a full evaluation of a single agent checkpoint.

    run_benchmark(config, n_episodes)
        Run all agent variants (Paper Table 1) as a single task.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

from celery import Task
from celery.exceptions import SoftTimeLimitExceeded

from tasks.celery_app import celery_app

logger = logging.getLogger(__name__)


class _BaseEvalTask(Task):
    """Base task that ensures environment cleanup on exit."""

    abstract = True

    def __init__(self) -> None:
        super().__init__()
        self._env = None

    def after_return(self, status, retval, task_id, args, kwargs, einfo) -> None:
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
            self._env = None


@celery_app.task(
    bind          = True,
    base          = _BaseEvalTask,
    name          = "tasks.evaluation_tasks.run_evaluation",
    queue         = "default",
    max_retries   = 0,
    track_started = True,
    ignore_result = False,
)
def run_evaluation(
    self:             Task,
    config:           Dict[str, Any],
    n_episodes:       int = 50,
    checkpoint_path:  Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run a full evaluation of a trained agent checkpoint.

    Parameters
    ----------
    config : dict
        Full application config (from config.yaml).
    n_episodes : int
        Number of evaluation episodes.  Default 50.
    checkpoint_path : str, optional
        Path to the checkpoint .zip file.  If None, evaluates the
        default checkpoint configured in config.yaml.

    Returns
    -------
    dict
        Full EvalResult dictionary with all paper metrics.
    """
    run_id     = (self.request.id or "local")[:8]
    start_time = time.monotonic()

    logger.info(
        "Evaluation task START — run_id=%s  episodes=%d  checkpoint=%s",
        run_id, n_episodes, checkpoint_path or "default",
    )

    self.update_state(
        state = "STARTED",
        meta  = {
            "run_id":     run_id,
            "n_episodes": n_episodes,
            "message":    "Loading agent and environment...",
        },
    )

    try:
        from pathlib import Path
        import numpy as np
        from envs.env_factory import make_env
        from agents.registry import AgentRegistry
        from evaluate import run_eval

        agent_cfg = config.get("agent", {})
        self._env = make_env(config, n_envs=1, mode="eval")
        agent     = AgentRegistry.build(self._env, agent_cfg)

        # Load checkpoint if provided
        if checkpoint_path and Path(checkpoint_path).exists():
            agent.model = agent.model.__class__.load(
                checkpoint_path, env=self._env,
            )
            logger.info("Loaded checkpoint: %s", checkpoint_path)

        self.update_state(
            state = "PROGRESS",
            meta  = {
                "run_id":     run_id,
                "n_episodes": n_episodes,
                "message":    f"Running {n_episodes} evaluation episodes...",
            },
        )

        result  = run_eval(agent=agent, n_episodes=n_episodes, env=self._env)
        elapsed = round(time.monotonic() - start_time, 2)

        logger.info(
            "Evaluation COMPLETE — run_id=%s  mean=%.4f  cvar_005=%.4f  elapsed=%.1fs",
            run_id,
            result.get("mean_reward", 0.0),
            result.get("cvar_005", 0.0),
            elapsed,
        )

        return {
            "status":   "completed",
            "run_id":   run_id,
            **result,
        }

    except SoftTimeLimitExceeded:
        logger.warning("Evaluation time limit reached — returning partial results.")
        return {
            "status":     "time_limit",
            "run_id":     run_id,
            "elapsed_s":  round(time.monotonic() - start_time, 1),
            "message":    "Evaluation stopped due to time limit.",
        }
    except Exception as exc:
        logger.error("Evaluation FAILED — %s", exc, exc_info=True)
        raise


@celery_app.task(
    bind          = True,
    base          = _BaseEvalTask,
    name          = "tasks.evaluation_tasks.run_benchmark",
    queue         = "default",
    max_retries   = 0,
    track_started = True,
    ignore_result = False,
)
def run_benchmark(
    self:       Task,
    config:     Dict[str, Any],
    n_episodes: int = 50,
) -> Dict[str, Any]:
    """
    Run the full benchmark comparison (Paper Table 1).

    Evaluates all agent variants: CVaR-PPO+EWC, Standard PPO,
    CVaR-only, EWC-only, and Random.

    Parameters
    ----------
    config : dict
        Full application config.
    n_episodes : int
        Episodes per agent variant.

    Returns
    -------
    dict
        ``{status, rows, elapsed_s}`` where rows is a list of BenchmarkRow dicts.
    """
    run_id     = (self.request.id or "local")[:8]
    start_time = time.monotonic()

    logger.info(
        "Benchmark task START — run_id=%s  episodes=%d",
        run_id, n_episodes,
    )

    self.update_state(
        state = "STARTED",
        meta  = {"run_id": run_id, "message": "Starting benchmark..."},
    )

    try:
        from evaluate import run_benchmark as _run_benchmark

        rows = _run_benchmark(config, n_episodes=n_episodes, seed=42)
        elapsed = round(time.monotonic() - start_time, 1)

        logger.info("Benchmark COMPLETE — run_id=%s  elapsed=%.1fs", run_id, elapsed)

        return {
            "status":    "completed",
            "run_id":    run_id,
            "rows":      rows,
            "elapsed_s": elapsed,
        }

    except SoftTimeLimitExceeded:
        logger.warning("Benchmark time limit reached.")
        return {
            "status":    "time_limit",
            "run_id":    run_id,
            "elapsed_s": round(time.monotonic() - start_time, 1),
        }
    except Exception as exc:
        logger.error("Benchmark FAILED — %s", exc, exc_info=True)
        raise
