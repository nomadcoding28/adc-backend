"""
CVaR Optimizer — Statistics & Evaluation Module
------------------------------------------------
Handles CVaR computation for EVALUATION and MONITORING.
Training-time CVaR loss lives in agents/cvar_ppo.py (CVaRComputer).

Used by:
  - evaluation/evaluator.py  → compute CVaR over test episodes
  - api/main.py              → serve CVaR metrics to dashboard
  - scripts/train_full.py    → log risk stats during training
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import deque
from loguru import logger


class CVaROptimizer:
    """
    CVaR statistics tracker for evaluation and monitoring.

    Usage:
        cvar = CVaROptimizer(alpha=0.05)
        for r in episode_rewards:
            cvar.update_reward_history(r)
        report = cvar.get_risk_report()
    """

    def __init__(self, alpha: float = 0.05, window_size: int = 10_000):
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0,1), got {alpha}")
        self.alpha = alpha
        self.window_size = window_size
        self._reward_history: deque = deque(maxlen=window_size)
        self._episode_count = 0
        logger.info(f"CVaROptimizer initialised | alpha={alpha}")

    def compute_cvar(self, rewards: np.ndarray, alpha: Optional[float] = None) -> float:
        a = alpha or self.alpha
        if len(rewards) == 0:
            return 0.0
        sorted_r = np.sort(rewards)
        cutoff = max(1, int(np.ceil(a * len(sorted_r))))
        return float(np.mean(sorted_r[:cutoff]))

    def compute_var(self, rewards: np.ndarray, alpha: Optional[float] = None) -> float:
        a = alpha or self.alpha
        if len(rewards) == 0:
            return 0.0
        return float(np.quantile(rewards, a))

    def tail_expectation(self, rewards: np.ndarray) -> Tuple[float, float, int]:
        var = self.compute_var(rewards)
        tail = rewards[rewards <= var]
        cvar = float(np.mean(tail)) if len(tail) > 0 else var
        return cvar, var, len(tail)

    def update_reward_history(self, reward: float):
        self._reward_history.append(reward)
        self._episode_count += 1

    def update_batch(self, rewards: List[float]):
        for r in rewards:
            self._reward_history.append(r)
        self._episode_count += len(rewards)

    def get_running_cvar(self) -> float:
        if len(self._reward_history) < max(10, int(1 / self.alpha)):
            return 0.0
        return self.compute_cvar(np.array(self._reward_history))

    def get_risk_report(self) -> Dict:
        if len(self._reward_history) < 5:
            return {"error": "Insufficient data", "n_samples": len(self._reward_history)}
        rewards = np.array(self._reward_history)
        cvar, var, n_tail = self.tail_expectation(rewards)
        return {
            "cvar": cvar,
            "var": var,
            "n_tail_samples": n_tail,
            "tail_fraction": n_tail / len(rewards),
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "min_reward": float(np.min(rewards)),
            "max_reward": float(np.max(rewards)),
            "median_reward": float(np.median(rewards)),
            "p5_reward":  float(np.percentile(rewards, 5)),
            "p10_reward": float(np.percentile(rewards, 10)),
            "p25_reward": float(np.percentile(rewards, 25)),
            "risk_ratio": cvar / (abs(np.mean(rewards)) + 1e-10),
            "alpha": self.alpha,
            "n_samples": len(rewards),
            "total_episodes": self._episode_count,
        }

    def compare_alpha_sensitivity(
        self,
        rewards: np.ndarray,
        alphas: Optional[List[float]] = None,
    ) -> Dict[float, Dict]:
        """CVaR at multiple α values — key ablation table for paper."""
        if alphas is None:
            alphas = [0.01, 0.05, 0.10, 0.20, 0.50]
        results = {}
        for a in alphas:
            results[a] = {
                "cvar": self.compute_cvar(rewards, alpha=a),
                "var": self.compute_var(rewards, alpha=a),
                "relative_to_mean": self.compute_cvar(rewards, alpha=a) / (
                    abs(np.mean(rewards)) + 1e-10
                ),
            }
        return results

    def compare_cvar_vs_standard(
        self,
        cvar_agent_rewards: np.ndarray,
        standard_agent_rewards: np.ndarray,
    ) -> Dict:
        """
        Head-to-head comparison: CVaR-PPO vs standard PPO.
        Key evaluation table for the paper.
        """
        def _stats(rewards):
            cvar, var, n_tail = self.tail_expectation(rewards)
            return {
                "mean": float(np.mean(rewards)),
                "std": float(np.std(rewards)),
                "cvar": cvar,
                "var": var,
                "min": float(np.min(rewards)),
                "success_rate": float(np.mean(rewards > 0)),
                "catastrophic_rate": float(
                    np.mean(rewards < np.percentile(rewards, 5))
                ),
            }
        cvar_s = _stats(cvar_agent_rewards)
        std_s = _stats(standard_agent_rewards)
        cvar_imp = ((cvar_s["cvar"] - std_s["cvar"]) / (abs(std_s["cvar"]) + 1e-10)) * 100
        mean_cost = ((cvar_s["mean"] - std_s["mean"]) / (abs(std_s["mean"]) + 1e-10)) * 100
        cat_red = ((std_s["catastrophic_rate"] - cvar_s["catastrophic_rate"]) / (std_s["catastrophic_rate"] + 1e-10)) * 100
        return {
            "cvar_ppo": cvar_s,
            "standard_ppo": std_s,
            "cvar_improvement_pct": cvar_imp,
            "mean_reward_cost_pct": mean_cost,
            "catastrophic_failure_reduction_pct": cat_red,
        }

    def reset(self):
        self._reward_history.clear()
        self._episode_count = 0