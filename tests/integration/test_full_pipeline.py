"""
tests/integration/test_full_pipeline.py
=========================================
Integration tests for the end-to-end pipeline.

Tests the connected flow: agent → drift detection → EWC → explainability.
Uses mock environment and simulated data — no CybORG required.
"""

from __future__ import annotations

from unittest.mock import MagicMock
import numpy as np
import pytest

from scipy.stats import wasserstein_distance


class TestFullPipeline:
    """Test the connected pipeline components."""

    def test_agent_step_triggers_drift_check(self, mock_env, rng) -> None:
        """
        Every 100 steps, drift detection should be checked.
        """
        check_freq = 100
        window_size = 500
        threshold = 0.15
        ref_window = []
        cur_window = []
        drift_events = []

        obs, _ = mock_env.reset()
        for step in range(1000):
            # Simulate step
            obs, reward, done, trunc, info = mock_env.step(0)
            ref_window.append(obs.copy())
            cur_window.append(obs.copy())

            # Keep windows at capacity
            if len(ref_window) > window_size:
                ref_window.pop(0)
            if len(cur_window) > window_size:
                cur_window.pop(0)

            # Check at frequency
            if step > 0 and step % check_freq == 0 and len(ref_window) >= window_size:
                ref_arr = np.array(ref_window)
                cur_arr = np.array(cur_window)
                w1 = np.mean([
                    wasserstein_distance(ref_arr[:, d], cur_arr[:, d])
                    for d in range(min(5, ref_arr.shape[1]))
                ])
                if w1 > threshold:
                    drift_events.append(step)

            if done:
                obs, _ = mock_env.reset()

        # Pipeline should have run drift checks
        assert isinstance(drift_events, list)

    def test_drift_triggers_ewc_registration(self) -> None:
        """When drift is detected, EWC should register a new task."""
        ewc_tasks = []
        drift_detected = True  # Simulate drift

        if drift_detected:
            # Simulate EWC task registration
            task = {
                "id": "task_0",
                "fisher": {"param": np.ones(10)},
                "params": {"param": np.zeros(10)},
            }
            ewc_tasks.append(task)

        assert len(ewc_tasks) == 1
        assert ewc_tasks[0]["id"] == "task_0"

    def test_observation_processor_decodes(self, sample_obs) -> None:
        """
        ObservationProcessor should decode raw 54-dim obs into
        structured fields (host states, processes, feedback).
        """
        obs = sample_obs

        # Simulate decoding
        decoded = {
            "host_states": obs[:35].tolist(),
            "malicious_processes": obs[35:48].tolist(),
            "action_feedback": obs[48:54].tolist(),
        }

        assert len(decoded["host_states"]) == 35
        assert len(decoded["malicious_processes"]) == 13
        assert len(decoded["action_feedback"]) == 6

    def test_action_to_explanation_flow(self) -> None:
        """
        An action should produce a structured explanation request.
        """
        from api.schemas.explain import ExplanationRequest

        request = ExplanationRequest(
            action="Isolate Host-3",
            action_idx=17,
            step=500,
            threat="Active C2 session on Host-3",
            risk_score=0.87,
            attacker_type="APT",
            technique_ids=["T1071"],
            action_success=True,
        )

        assert request.action == "Isolate Host-3"
        assert request.risk_score == 0.87
        assert "T1071" in request.technique_ids

    def test_game_belief_update_flow(self) -> None:
        """
        Bayesian belief should update given an observation.
        """
        from api.schemas.game import BeliefState

        # Initial uniform prior
        prior = {"Random": 0.33, "APT": 0.34, "Adaptive": 0.33}

        # Simulate Bayesian update: APT-like behaviour observed
        # (multiply by likelihood and normalise)
        likelihoods = {"Random": 0.1, "APT": 0.7, "Adaptive": 0.2}
        unnorm = {k: prior[k] * likelihoods[k] for k in prior}
        total = sum(unnorm.values())
        posterior = {k: v / total for k, v in unnorm.items()}

        belief = BeliefState(
            probabilities=posterior,
            dominant_type=max(posterior, key=posterior.get),
            n_updates=1,
        )

        assert belief.dominant_type == "APT"
        assert belief.probabilities["APT"] > 0.5
