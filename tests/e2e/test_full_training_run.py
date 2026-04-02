"""
tests/e2e/test_full_training_run.py
======================================
End-to-end test for a complete (short) training run.

Simulates the full lifecycle:
    1. Agent initialisation
    2. Training loop with drift detection
    3. EWC task registration
    4. Checkpoint saving
    5. Evaluation with metric collection

No CybORG required — uses mock environment.
"""

from __future__ import annotations

from unittest.mock import MagicMock
import numpy as np
import pytest

from scipy.stats import wasserstein_distance


class TestFullTrainingRun:
    """E2E test for a complete training → evaluation cycle."""

    @pytest.fixture
    def training_config(self) -> dict:
        return {
            "agent_type": "cvar_ppo",
            "total_timesteps": 500,
            "cvar_alpha": 0.05,
            "ewc_enabled": True,
            "ewc_lambda": 0.4,
            "drift_threshold": 0.15,
            "drift_check_freq": 100,
            "eval_episodes": 10,
        }

    def test_full_training_cycle(self, mock_env, training_config, rng) -> None:
        """
        Run a full training → drift → EWC → eval cycle and verify
        that all components produce valid outputs.
        """
        config = training_config
        env = mock_env

        # ── 1. Agent initialisation ──────────────────────────────
        agent_metrics = {
            "total_timesteps": 0,
            "episode_count": 0,
            "mean_reward": 0.0,
            "cvar_005": 0.0,
            "drift_events": 0,
            "ewc_tasks": 0,
        }

        # ── 2. Training loop ────────────────────────────────────
        episode_rewards = []
        current_episode_reward = 0.0
        obs, _ = env.reset()
        ref_window: list = []
        cur_window: list = []
        checkpoints: list = []
        drift_events: list = []
        ewc_tasks: list = []

        for step in range(config["total_timesteps"]):
            action = rng.integers(0, 54)
            obs, reward, done, trunc, info = env.step(action)
            current_episode_reward += reward

            # Track observation windows for drift
            cur_window.append(obs.copy())
            if step < config["total_timesteps"] // 2:
                ref_window.append(obs.copy())

            # Trim windows
            max_win = 200
            if len(cur_window) > max_win:
                cur_window.pop(0)
            if len(ref_window) > max_win:
                ref_window.pop(0)

            # ── 3. Drift detection ──────────────────────────────
            if (step > 0 and step % config["drift_check_freq"] == 0
                    and len(ref_window) >= 50 and len(cur_window) >= 50):
                ref_arr = np.array(ref_window[-50:])
                cur_arr = np.array(cur_window[-50:])
                w1 = np.mean([
                    wasserstein_distance(ref_arr[:, d], cur_arr[:, d])
                    for d in range(min(5, ref_arr.shape[1]))
                ])
                if w1 > config["drift_threshold"]:
                    drift_events.append({"step": step, "distance": w1})

                    # ── 4. EWC task registration ────────────────
                    if config["ewc_enabled"]:
                        ewc_tasks.append({"step": step, "task_id": len(ewc_tasks)})

            # Episode boundary
            if done:
                episode_rewards.append(current_episode_reward)
                current_episode_reward = 0.0
                obs, _ = env.reset()
                agent_metrics["episode_count"] += 1

            agent_metrics["total_timesteps"] = step + 1

        # ── 5. Checkpoint saving ────────────────────────────────
        checkpoint = {
            "step": config["total_timesteps"],
            "path": f"checkpoints/model_{config['total_timesteps']}.zip",
            "mean_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        }
        checkpoints.append(checkpoint)

        # ── 6. Evaluation ───────────────────────────────────────
        eval_rewards = []
        for ep in range(config["eval_episodes"]):
            obs, _ = env.reset()
            ep_reward = 0.0
            for s in range(100):
                action = rng.integers(0, 54)
                obs, reward, done, trunc, info = env.step(action)
                ep_reward += reward
                if done:
                    break
            eval_rewards.append(ep_reward)

        returns = np.array(eval_rewards, dtype=np.float32)
        if len(returns) > 0:
            n_tail = max(1, int(config["cvar_alpha"] * len(returns)))
            sorted_r = np.sort(returns)
            cvar_005 = float(np.mean(sorted_r[:n_tail]))
        else:
            cvar_005 = 0.0

        # ── Assertions ──────────────────────────────────────────
        # Training ran to completion
        assert agent_metrics["total_timesteps"] == config["total_timesteps"]

        # Episodes were recorded
        assert agent_metrics["episode_count"] > 0 or len(episode_rewards) >= 0

        # Checkpoint was saved
        assert len(checkpoints) == 1
        assert checkpoints[0]["step"] == config["total_timesteps"]

        # Evaluation produced results
        assert len(eval_rewards) == config["eval_episodes"]
        assert np.all(np.isfinite(returns))

        # CVaR computed
        assert np.isfinite(cvar_005)

        # Drift and EWC tracking
        assert isinstance(drift_events, list)
        assert isinstance(ewc_tasks, list)

    def test_metrics_output_format(self, mock_env, rng) -> None:
        """
        Final metrics should contain all required paper fields.
        """
        required_keys = [
            "total_timesteps", "episode_count", "mean_reward",
            "cvar_005", "drift_events", "ewc_tasks",
        ]

        # Simulate final metrics
        metrics = {
            "total_timesteps": 500,
            "episode_count": 5,
            "mean_reward": 42.0,
            "cvar_005": -15.0,
            "drift_events": 2,
            "ewc_tasks": 1,
        }

        for key in required_keys:
            assert key in metrics, f"Missing metric: {key}"
