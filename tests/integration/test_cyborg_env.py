"""
tests/integration/test_cyborg_env.py
======================================
Integration tests for the CybORG environment wrapper.

Uses the mock environment — CybORG is NOT required.
"""

from __future__ import annotations

from unittest.mock import MagicMock
import numpy as np
import pytest


class TestCybORGWrapper:
    """Test the environment wrapper lifecycle and data shapes."""

    def test_observation_shape(self, mock_env) -> None:
        """Observations should have shape (54,)."""
        obs, info = mock_env.reset()
        assert obs.shape == (54,), f"Expected (54,), got {obs.shape}"

    def test_observation_dtype(self, mock_env) -> None:
        """Observations should be float32."""
        obs, info = mock_env.reset()
        assert obs.dtype == np.float32

    def test_action_space_size(self, mock_env) -> None:
        """Action space should have 54 actions."""
        assert mock_env.action_space.n == 54

    def test_step_returns_correct_signature(self, mock_env) -> None:
        """step() should return (obs, reward, terminated, truncated, info)."""
        mock_env.reset()
        result = mock_env.step(0)
        assert len(result) == 5, f"Expected 5-tuple from step(), got {len(result)}"

        obs, reward, terminated, truncated, info = result
        assert obs.shape == (54,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_reset_returns_obs_and_info(self, mock_env) -> None:
        """reset() should return (obs, info) tuple."""
        result = mock_env.reset()
        assert len(result) == 2
        obs, info = result
        assert obs.shape == (54,)
        assert isinstance(info, dict)

    def test_close_is_callable(self, mock_env) -> None:
        """close() should be callable without error."""
        mock_env.close()
        mock_env.close.assert_called_once()

    def test_rewards_are_bounded(self, mock_env) -> None:
        """Step rewards should be finite."""
        mock_env.reset()
        for _ in range(10):
            obs, reward, done, trunc, info = mock_env.step(0)
            assert np.isfinite(reward), f"Reward {reward} is not finite."
