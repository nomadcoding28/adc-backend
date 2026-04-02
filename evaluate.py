"""
evaluate.py
===========
ACD Framework — Evaluation and Benchmark Runner.

Runs a full evaluation suite against a trained checkpoint and produces
all metrics needed for the paper (Table 1 + Table 2).

Usage
-----
    # Evaluate a single checkpoint
    python evaluate.py --checkpoint data/checkpoints/cvar_ppo_final.zip

    # Full benchmark (all agent variants — takes ~30 min)
    python evaluate.py --benchmark --n-episodes 50

    # α-sensitivity table (Paper Table 2)
    python evaluate.py --checkpoint data/checkpoints/best.zip --alpha-sweep

    # Export results to CSV for the paper
    python evaluate.py --benchmark --export data/paper_results.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from monitoring.structlog_config import configure_logging
from utils.seed import set_seed

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ACD Framework — Evaluation Script")

    p.add_argument("--checkpoint",  default=None,           help="Path to checkpoint .zip")
    p.add_argument("--config",      default="config.yaml",  help="Config YAML path")
    p.add_argument("--n-episodes",  type=int, default=50,   help="Evaluation episodes")
    p.add_argument("--scenario",    default="scenario2")
    p.add_argument("--benchmark",   action="store_true",    help="Run full benchmark table")
    p.add_argument("--alpha-sweep", action="store_true",    help="Run α-sensitivity sweep")
    p.add_argument("--export",      default=None,           help="Export results to CSV path")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--verbose",     action="store_true")

    return p.parse_args()


# ── Core evaluation function ────────────────────────────────────────────────

def run_eval(
    agent,
    n_episodes: int = 50,
    env         = None,
    seed:       int = 42,
) -> dict:
    """
    Run n_episodes and compute all paper metrics.

    Parameters
    ----------
    agent       : BaseAgent
    n_episodes  : int
    env         : Gym env, optional. Uses agent.env if None.
    seed        : int

    Returns
    -------
    dict
        Full evaluation result dict matching EvalResult schema.
    """
    eval_env = env or agent.env
    rewards: list[float] = []
    start = time.monotonic()

    for ep in range(n_episodes):
        obs, _ = eval_env.reset(seed=seed + ep)
        ep_r   = 0.0
        done   = False
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = eval_env.step(int(action))
            ep_r += r
            done = term or trunc
        rewards.append(ep_r)

    arr = np.array(rewards)

    def cvar(alpha: float) -> float:
        n_tail = max(1, int(alpha * len(arr)))
        return float(np.mean(np.sort(arr)[:n_tail]))

    return {
        "n_episodes":         n_episodes,
        "mean_reward":        round(float(np.mean(arr)),  4),
        "std_reward":         round(float(np.std(arr)),   4),
        "min_reward":         round(float(np.min(arr)),   4),
        "max_reward":         round(float(np.max(arr)),   4),
        "cvar_001":           round(cvar(0.01), 4),
        "cvar_005":           round(cvar(0.05), 4),
        "cvar_010":           round(cvar(0.10), 4),
        "cvar_020":           round(cvar(0.20), 4),
        "cvar_050":           round(cvar(0.50), 4),
        "success_rate":       round(float(np.mean(arr > 0)), 4),
        "catastrophic_rate":  round(float(np.mean(arr < -10)), 4),
        "elapsed_s":          round(time.monotonic() - start, 2),
    }


# ── Benchmark table ─────────────────────────────────────────────────────────

def run_benchmark(
    config:     dict,
    n_episodes: int = 50,
    seed:       int = 42,
) -> list[dict]:
    """Run all agent variants and return a list of result rows."""
    from envs.env_factory import make_env
    from agents.registry import AgentRegistry

    variants = [
        ("CVaR-PPO + EWC (Ours)",  "cvar_ppo",     True,  True),
        ("Standard PPO",           "standard_ppo", False, False),
        ("PPO + CVaR (no EWC)",    "cvar_ppo_no_ewc", True, False),
        ("PPO + EWC (no CVaR)",    "ppo_ewc_only", False, True),
        ("Random Agent",           "random",       False, False),
    ]

    rows = []
    for label, agent_type, use_cvar, use_ewc in variants:
        logger.info("Evaluating: %s", label)
        try:
            variant_cfg = dict(config.get("agent", {}))
            variant_cfg["agent_type"]            = agent_type
            variant_cfg.setdefault("cvar", {})["enabled"] = use_cvar
            variant_cfg.setdefault("ewc",  {})["enabled"] = use_ewc

            env    = make_env(config, n_envs=1, mode="eval")
            agent  = AgentRegistry.build(env, variant_cfg)
            result = run_eval(agent, n_episodes=n_episodes, seed=seed)

            rows.append({
                "agent":             label,
                "mean_reward":       result["mean_reward"],
                "cvar_005":          result["cvar_005"],
                "success_rate":      result["success_rate"],
                "catastrophic_rate": result["catastrophic_rate"],
                "is_ours":           agent_type == "cvar_ppo",
            })
            logger.info(
                "  Mean=%.4f  CVaR=%.4f  Success=%.1f%%  Catastro=%.1f%%",
                result["mean_reward"], result["cvar_005"],
                result["success_rate"] * 100, result["catastrophic_rate"] * 100,
            )
        except Exception as exc:
            logger.error("  FAILED: %s", exc)
            rows.append({"agent": label, "error": str(exc)})

    return rows


# ── α-sensitivity sweep ─────────────────────────────────────────────────────

def alpha_sweep(agent, n_episodes: int = 50, seed: int = 42) -> list[dict]:
    """Run eval and compute CVaR for multiple α values — Paper Table 2."""
    result = run_eval(agent, n_episodes=n_episodes, seed=seed)
    alphas = [0.01, 0.05, 0.10, 0.20, 0.50]

    return [
        {
            "alpha":           a,
            "cvar":            result.get(f"cvar_{int(a*100):03d}", result["cvar_005"]),
            "interpretation":  _alpha_label(a),
            "is_default":      abs(a - 0.05) < 1e-6,
        }
        for a in alphas
    ]


def _alpha_label(alpha: float) -> str:
    if alpha <= 0.01:  return "Worst 1%  — extremely risk-averse"
    if alpha <= 0.05:  return "Worst 5%  — default setting"
    if alpha <= 0.10:  return "Worst 10% — moderately risk-averse"
    if alpha <= 0.20:  return "Worst 20% — mildly risk-averse"
    return "Worst 50% — near risk-neutral"


# ── CSV export ───────────────────────────────────────────────────────────────

def export_csv(rows: list[dict], path: str) -> None:
    """Write result rows to a CSV file."""
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Results exported to: %s", path)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    configure_logging(level="DEBUG" if args.verbose else "INFO")
    set_seed(args.seed)

    from utils.config_loader import load_config
    config = load_config(args.config)

    if args.benchmark:
        logger.info("Running full benchmark (%d episodes per agent)...", args.n_episodes)
        rows = run_benchmark(config, n_episodes=args.n_episodes, seed=args.seed)

        print("\n" + "="*80)
        print("BENCHMARK TABLE — Paper Table 1")
        print("="*80)
        header = f"{'Agent':<30} {'Mean Rwd':>10} {'CVaR 0.05':>10} {'Success':>8} {'Catast':>8}"
        print(header)
        print("-"*80)
        for row in rows:
            if "error" in row:
                print(f"  {row['agent']:<28} ERROR: {row['error']}")
                continue
            marker = " ★" if row.get("is_ours") else "  "
            print(
                f"{marker}{row['agent']:<28} "
                f"{row['mean_reward']:>10.4f} "
                f"{row['cvar_005']:>10.4f} "
                f"{row['success_rate']:>7.1%} "
                f"{row['catastrophic_rate']:>7.1%}"
            )
        print("="*80)

        if args.export:
            export_csv(rows, args.export)

    elif args.checkpoint:
        from envs.env_factory import make_env
        from agents.registry import AgentRegistry
        from pathlib import Path

        logger.info("Evaluating checkpoint: %s", args.checkpoint)
        agent_cfg = config.get("agent", {})
        env   = make_env(config, n_envs=1, mode="eval")
        agent = AgentRegistry.build(env, agent_cfg)

        if Path(args.checkpoint).exists():
            agent.model = agent.model.__class__.load(args.checkpoint, env=env)

        result = run_eval(agent, n_episodes=args.n_episodes, seed=args.seed)

        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        for k, v in result.items():
            print(f"  {k:<25} {v}")
        print("="*60)

        if args.alpha_sweep:
            rows = alpha_sweep(agent, n_episodes=args.n_episodes, seed=args.seed)
            print("\nα-SENSITIVITY TABLE — Paper Table 2")
            print("-"*55)
            for row in rows:
                marker = " ← default" if row["is_default"] else ""
                print(f"  α={row['alpha']:.2f}  CVaR={row['cvar']:>8.4f}  {row['interpretation']}{marker}")
            print("-"*55)

            if args.export:
                export_csv(rows, args.export)

    else:
        logger.error("Specify --checkpoint <path> or --benchmark")
        sys.exit(1)


if __name__ == "__main__":
    main()