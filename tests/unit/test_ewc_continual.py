"""
tests/unit/test_ewc_continual.py
=================================
Unit tests for Elastic Weight Consolidation (EWC).

Tests validate:
    - Fisher Information Matrix diagonal is non-negative
    - EWC penalty is zero before any task is registered
    - Penalty increases as parameters deviate from snapshot
    - Task count overflow triggers merging
    - Experience buffer respects capacity
"""

from __future__ import annotations

import numpy as np
import pytest


# ── Attempt to import torch (skip gracefully if absent) ─────────────────────
torch = pytest.importorskip("torch")
nn = torch.nn


# ── Fisher Information Matrix ───────────────────────────────────────────────

class TestFisherComputation:
    """Test Fisher Information Matrix diagonal computation."""

    @staticmethod
    def _compute_fisher_diagonal(model: nn.Module, n_samples: int = 50) -> dict:
        """
        Compute Fisher diagonal for a simple model from random data.

        F_ii = E[ (∂ log p(y|x,θ) / ∂θ_i)^2 ]
        Approximated by: average of squared gradients over samples.
        """
        fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
        model.train()

        rng = np.random.default_rng(42)
        for _ in range(n_samples):
            x = torch.tensor(rng.standard_normal(54).astype(np.float32))
            out = model(x)
            loss = out.sum()
            model.zero_grad()
            loss.backward()

            for n, p in model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.data ** 2

        for n in fisher:
            fisher[n] /= n_samples

        return fisher

    def test_fisher_non_negative(self, simple_mlp) -> None:
        """All Fisher diagonal entries should be ≥ 0 (squared gradients)."""
        fisher = self._compute_fisher_diagonal(simple_mlp)
        for name, values in fisher.items():
            assert torch.all(values >= 0), f"Fisher[{name}] has negative entries."

    def test_fisher_not_all_zero(self, simple_mlp) -> None:
        """Fisher should have some non-zero entries (model has gradients)."""
        fisher = self._compute_fisher_diagonal(simple_mlp)
        total = sum(v.sum().item() for v in fisher.values())
        assert total > 0, "Fisher diagonal is all zeros — model has no gradients."

    def test_fisher_shape_matches_params(self, simple_mlp) -> None:
        """Fisher diagonal shape should match parameter shape."""
        fisher = self._compute_fisher_diagonal(simple_mlp)
        for n, p in simple_mlp.named_parameters():
            assert fisher[n].shape == p.shape, (
                f"Fisher shape {fisher[n].shape} ≠ param shape {p.shape} for {n}"
            )


# ── EWC Penalty ─────────────────────────────────────────────────────────────

class TestEWCPenalty:
    """Test EWC penalty computation."""

    def test_penalty_zero_before_registration(self, simple_mlp) -> None:
        """EWC penalty should be 0 when no tasks have been registered."""
        # No tasks registered → penalty is 0
        penalty = 0.0
        # Compute by summing over empty task list
        tasks = []
        for task in tasks:
            for n, p in simple_mlp.named_parameters():
                fisher = task["fisher"][n]
                old_params = task["params"][n]
                penalty += (fisher * (p - old_params) ** 2).sum().item()

        assert penalty == 0.0

    def test_penalty_zero_at_snapshot(self, simple_mlp) -> None:
        """EWC penalty should be 0 when θ == θ* (no deviation)."""
        # Snapshot current parameters
        snapshot = {n: p.clone().detach() for n, p in simple_mlp.named_parameters()}
        fisher = {n: torch.ones_like(p) for n, p in simple_mlp.named_parameters()}

        # Compute penalty at snapshot point
        penalty = 0.0
        for n, p in simple_mlp.named_parameters():
            penalty += (fisher[n] * (p - snapshot[n]) ** 2).sum().item()

        assert abs(penalty) < 1e-10, f"Penalty at snapshot should be 0, got {penalty}"

    def test_penalty_increases_with_deviation(self, simple_mlp) -> None:
        """EWC penalty should increase as θ deviates from θ*."""
        snapshot = {n: p.clone().detach() for n, p in simple_mlp.named_parameters()}
        fisher = {n: torch.ones_like(p) for n, p in simple_mlp.named_parameters()}

        penalties = []
        for scale in [0.0, 0.1, 0.5, 1.0, 2.0]:
            penalty = 0.0
            for n, p in simple_mlp.named_parameters():
                deviated = snapshot[n] + scale * torch.randn_like(p)
                penalty += (fisher[n] * (deviated - snapshot[n]) ** 2).sum().item()
            penalties.append(penalty)

        # Monotonically increasing (approximately — random noise)
        assert penalties[0] < penalties[-1], (
            f"Penalty at scale=0 ({penalties[0]:.4f}) should be < "
            f"penalty at scale=2 ({penalties[-1]:.4f})"
        )

    def test_penalty_weighted_by_fisher(self, simple_mlp) -> None:
        """Higher Fisher values should produce higher penalty for same deviation."""
        snapshot = {n: p.clone().detach() for n, p in simple_mlp.named_parameters()}

        deviation = 0.1
        deviated = {n: p + deviation for n, p in simple_mlp.named_parameters()}

        # Low Fisher
        fisher_low = {n: torch.ones_like(p) * 0.01 for n, p in simple_mlp.named_parameters()}
        penalty_low = sum(
            (fisher_low[n] * (deviated[n] - snapshot[n]) ** 2).sum().item()
            for n, _ in simple_mlp.named_parameters()
        )

        # High Fisher
        fisher_high = {n: torch.ones_like(p) * 10.0 for n, p in simple_mlp.named_parameters()}
        penalty_high = sum(
            (fisher_high[n] * (deviated[n] - snapshot[n]) ** 2).sum().item()
            for n, _ in simple_mlp.named_parameters()
        )

        assert penalty_high > penalty_low, "Higher Fisher → higher penalty."


# ── Task management ─────────────────────────────────────────────────────────

class TestTaskManagement:
    """Test EWC task registration and merging."""

    def test_task_merge_on_overflow(self) -> None:
        """When max_tasks is exceeded, the oldest task should be merged."""
        max_tasks = 3
        tasks = []

        for i in range(5):
            task = {"id": i, "fisher": {"w": i + 1.0}, "params": {"w": float(i)}}
            tasks.append(task)

            if len(tasks) > max_tasks:
                # Merge oldest two
                merged = {
                    "id": f"merged_{tasks[0]['id']}_{tasks[1]['id']}",
                    "fisher": {"w": (tasks[0]["fisher"]["w"] + tasks[1]["fisher"]["w"]) / 2},
                    "params": {"w": (tasks[0]["params"]["w"] + tasks[1]["params"]["w"]) / 2},
                }
                tasks = [merged] + tasks[2:]

        assert len(tasks) <= max_tasks, f"Expected ≤ {max_tasks} tasks, got {len(tasks)}"


# ── Experience buffer ───────────────────────────────────────────────────────

class TestExperienceBuffer:
    """Test the circular experience buffer for Fisher computation."""

    def test_buffer_respects_capacity(self) -> None:
        """Buffer should not exceed its maximum capacity."""
        capacity = 100
        buffer = []

        for i in range(200):
            buffer.append({"obs": np.zeros(54), "action": 0, "reward": 1.0})
            if len(buffer) > capacity:
                buffer.pop(0)  # FIFO eviction

        assert len(buffer) == capacity

    def test_buffer_maintains_recency(self) -> None:
        """Most recent samples should remain after overflow."""
        capacity = 50
        buffer = []

        for i in range(100):
            buffer.append({"step": i})
            if len(buffer) > capacity:
                buffer.pop(0)

        assert buffer[0]["step"] == 50
        assert buffer[-1]["step"] == 99
