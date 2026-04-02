"""
tests/conftest.py
==================
Shared pytest fixtures for the ACD Framework test suite.

All fixtures avoid importing CybORG (not installed) — they create
mock objects and synthetic data for isolated unit testing.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

logger = logging.getLogger(__name__)

# ── Reproducible RNG ────────────────────────────────────────────────────────

@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded random number generator for deterministic tests."""
    return np.random.default_rng(42)


# ── Configuration fixtures ──────────────────────────────────────────────────

@pytest.fixture
def mock_config() -> Dict[str, Any]:
    """Minimal mock config matching config.yaml structure."""
    return {
        "agent": {
            "agent_type": "cvar_ppo",
            "total_timesteps": 10_000,
            "learning_rate": 3e-4,
            "n_steps": 128,
            "batch_size": 64,
            "n_epochs": 4,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "device": "cpu",
            "cvar": {
                "enabled": True,
                "alpha": 0.05,
                "weight": 0.30,
            },
            "ewc": {
                "enabled": True,
                "lambda_": 0.4,
                "max_tasks": 5,
                "fisher_n_samples": 200,
            },
        },
        "environment": {
            "scenario": "scenario2",
            "obs_dim": 54,
            "action_dim": 54,
            "max_steps": 100,
        },
        "drift": {
            "method": "wasserstein",
            "threshold": 0.15,
            "window_size": 500,
            "check_frequency": 100,
            "cooldown_steps": 200,
        },
        "game": {
            "attacker_types": ["Random", "APT", "Adaptive"],
            "n_red_actions": 4,
        },
        "llm": {
            "model": "gpt-4o-mini",
            "temperature": 0.1,
            "max_tokens": 500,
        },
        "knowledge_graph": {
            "uri": "bolt://localhost:7687",
            "user": "neo4j",
            "password": "test",
        },
    }


# ── Observation / action fixtures ───────────────────────────────────────────

@pytest.fixture
def sample_obs(rng) -> np.ndarray:
    """Single 54-dim observation vector."""
    return rng.standard_normal(54).astype(np.float32)


@pytest.fixture
def sample_obs_batch(rng) -> np.ndarray:
    """Batch of 100 observations, shape (100, 54)."""
    return rng.standard_normal((100, 54)).astype(np.float32)


@pytest.fixture
def sample_returns(rng) -> np.ndarray:
    """200 sample episode returns (mix of good and bad)."""
    good = rng.normal(loc=40.0, scale=15.0, size=180)
    bad  = rng.normal(loc=-30.0, scale=10.0, size=20)
    return np.concatenate([good, bad]).astype(np.float32)


@pytest.fixture
def sample_rewards(rng) -> np.ndarray:
    """500 step-level rewards."""
    return rng.normal(loc=0.5, scale=1.0, size=500).astype(np.float32)


# ── Mock environment ────────────────────────────────────────────────────────

@pytest.fixture
def mock_env():
    """
    Mock Gymnasium-compatible environment.

    Provides:
        - observation_space.shape = (54,)
        - action_space.n = 54
        - reset() / step() with synthetic data
    """
    env = MagicMock()

    # Spaces
    obs_space = MagicMock()
    obs_space.shape = (54,)
    obs_space.dtype = np.float32
    env.observation_space = obs_space

    act_space = MagicMock()
    act_space.n = 54
    env.action_space = act_space

    # Reset
    rng = np.random.default_rng(42)
    def _reset(*args, **kwargs):
        return rng.standard_normal(54).astype(np.float32), {}

    env.reset = MagicMock(side_effect=_reset)

    # Step
    def _step(action):
        obs = rng.standard_normal(54).astype(np.float32)
        reward = float(rng.normal(0.5, 1.0))
        terminated = rng.random() < 0.01
        truncated = False
        info = {"step": 0}
        return obs, reward, terminated, truncated, info

    env.step = MagicMock(side_effect=_step)
    env.close = MagicMock()

    return env


# ── Mock agent ──────────────────────────────────────────────────────────────

@pytest.fixture
def mock_agent():
    """Mock ACD agent with common interface methods."""
    agent = MagicMock()
    agent.is_training = False
    agent.total_timesteps_trained = 0
    agent.get_metrics.return_value = {
        "mean_reward": 42.0,
        "cvar_005": -15.0,
        "cvar_001": -28.0,
        "cvar_010": -10.0,
        "catastrophic_rate": 0.05,
        "success_rate": 0.85,
        "total_timesteps": 10000,
        "episode_count": 100,
    }
    agent.predict.return_value = (np.array([0]), None)
    return agent


# ── Mock Neo4j client ───────────────────────────────────────────────────────

@pytest.fixture
def mock_kg_client():
    """Mock Neo4j client that returns empty results."""
    client = MagicMock()
    client.execute_query.return_value = []
    client.get_stats.return_value = {
        "n_cves": 0,
        "n_techniques": 0,
        "n_tactics": 0,
        "n_hosts": 0,
    }
    client.close = MagicMock()
    return client


# ── Mock LLM client ─────────────────────────────────────────────────────────

@pytest.fixture
def mock_llm_client():
    """Mock LLM client returning canned responses."""
    client = MagicMock()
    client.model = "gpt-4o-mini-mock"

    response = MagicMock()
    response.content = "Mock LLM response for testing."
    response.prompt_tokens = 50
    response.latency_s = 0.1
    client.chat.return_value = response

    return client


# ── Torch fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def simple_mlp():
    """Create a simple 2-layer MLP for testing EWC."""
    try:
        import torch
        import torch.nn as nn

        model = nn.Sequential(
            nn.Linear(54, 64),
            nn.ReLU(),
            nn.Linear(64, 54),
        )
        return model
    except ImportError:
        pytest.skip("PyTorch not installed")
