"""
Elastic Weight Consolidation (EWC)
====================================
Novelty 1 — Continual Learning component.

Prevents catastrophic forgetting when the agent adapts to new
attack distributions after concept drift is detected.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THE MATH
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

When the agent finishes learning task A (old attack distribution),
EWC records:
  θ*_A  = current optimal parameters
  F_A   = diagonal Fisher information matrix

Fisher information for parameter θᵢ:
  Fᵢ = E[ (∂ log π(a|s) / ∂θᵢ)² ]

This measures how much the policy's action distribution
changes if we perturb θᵢ — high Fisher = this parameter
matters a lot for the current task.

When learning task B (new attack distribution), the loss becomes:
  L_B_EWC = L_B  +  (λ/2) · Σᵢ Fᵢ · (θᵢ - θ*_Aᵢ)²

The EWC term is a quadratic "memory" anchored at θ*_A,
weighted by how important each parameter was (Fᵢ).

For K previous tasks:
  L_EWC = Σₖ (λ/2) · Σᵢ Fᵢ^k · (θᵢ - θ*ᵢ^k)²

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FORGETTING METRIC (for paper evaluation)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

After learning task B, we measure forgetting on task A:
  Forget_A = R_A_before - R_A_after_B

  where R_A = reward on task A evaluation episodes

EWC should keep Forget_A close to 0 even as the agent
adapts to task B.

Reference: Kirkpatrick et al., 2017
"Overcoming catastrophic forgetting in neural networks"
PNAS 114(13): 3521-3526
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque
from loguru import logger


# ══════════════════════════════════════════════════════════════
#  Experience Buffer — feeds Fisher computation
# ══════════════════════════════════════════════════════════════

class ExperienceBuffer:
    """
    Circular buffer that stores (obs, action) transitions
    from the PPO rollout buffer.

    This is the "dataloader" EWC uses to estimate the Fisher
    information matrix. It must be populated during training
    BEFORE drift is detected, so it contains samples from the
    task we want to remember.

    Usage:
        buffer = ExperienceBuffer(capacity=2000)
        # Inside SB3 callback, each step:
        buffer.add(obs, action)
        # On drift:
        ewc.register_task(buffer)
    """

    def __init__(self, capacity: int = 2000):
        self.capacity = capacity
        self._obs:     deque = deque(maxlen=capacity)
        self._actions: deque = deque(maxlen=capacity)
        self._count = 0

    def add(self, obs: np.ndarray, action: int):
        """Add a single (obs, action) transition."""
        self._obs.append(obs.copy() if isinstance(obs, np.ndarray) else obs)
        self._actions.append(int(action))
        self._count += 1

    def add_batch(self, obs_batch: np.ndarray, action_batch: np.ndarray):
        """Add a batch of transitions (e.g. from SB3 rollout buffer)."""
        for obs, action in zip(obs_batch, action_batch):
            self.add(obs, int(action))

    def sample(
        self, n: int, device: str = "cpu"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample n (obs, action) pairs as tensors.

        Returns:
            obs_tensor:    FloatTensor of shape (n, obs_dim)
            action_tensor: LongTensor of shape (n,)
        """
        n = min(n, len(self))
        indices = np.random.choice(len(self), n, replace=False)

        obs_list     = list(self._obs)
        action_list  = list(self._actions)

        obs_arr    = np.stack([obs_list[i] for i in indices])
        action_arr = np.array([action_list[i] for i in indices])

        return (
            torch.FloatTensor(obs_arr).to(device),
            torch.LongTensor(action_arr).to(device),
        )

    def iter_batches(
        self, batch_size: int = 32, device: str = "cpu"
    ):
        """Iterate over all stored data in batches."""
        obs_list    = list(self._obs)
        action_list = list(self._actions)
        n = len(obs_list)

        indices = np.random.permutation(n)
        for start in range(0, n, batch_size):
            batch_idx = indices[start : start + batch_size]
            obs_arr    = np.stack([obs_list[i] for i in batch_idx])
            action_arr = np.array([action_list[i] for i in batch_idx])
            yield (
                torch.FloatTensor(obs_arr).to(device),
                torch.LongTensor(action_arr).to(device),
            )

    def __len__(self) -> int:
        return len(self._obs)

    @property
    def total_added(self) -> int:
        return self._count

    def is_ready(self, min_samples: int = 100) -> bool:
        return len(self) >= min_samples

    def clear(self):
        self._obs.clear()
        self._actions.clear()


# ══════════════════════════════════════════════════════════════
#  EWC — Elastic Weight Consolidation
# ══════════════════════════════════════════════════════════════

class ElasticWeightConsolidation:
    """
    EWC for continual learning in the ACD PPO agent.

    Lifecycle:
      1. Agent trains on task 0 (baseline attack distribution)
      2. Drift detected → call register_task(experience_buffer)
         - Saves θ*₀ (current optimal params)
         - Computes F₀ (Fisher information)
      3. Agent adapts to task 1 (new attack distribution)
         - EWC penalty in CVaRPPO.train() prevents forgetting task 0
      4. Another drift → register_task() again
         - Now has [task_0, task_1] in memory
         - Penalty grows: anchors agent to BOTH previous tasks

    Usage:
        ewc = ElasticWeightConsolidation(policy, lambda_ewc=0.4)
        # ... agent trains ...
        ewc.register_task(experience_buffer)   # on drift
        # penalty() is called automatically inside CVaRPPO.train()
    """

    def __init__(
        self,
        policy: nn.Module,
        lambda_ewc: float = 0.4,
        max_tasks: int = 5,
        fisher_cap: float = 1e4,
        device: str = "cpu",
    ):
        """
        Args:
            policy:      SB3 ActorCriticPolicy (or any nn.Module)
            lambda_ewc:  Regularisation strength. Higher = less forgetting
                         but slower adaptation to new attacks.
                         Recommended range: 0.1 – 1.0
            max_tasks:   Max number of tasks to remember. When exceeded,
                         oldest task is merged into the next (online EWC).
            fisher_cap:  Maximum Fisher value per parameter — prevents
                         numerical instability when a few samples dominate.
            device:      torch device string
        """
        self.policy     = policy
        self.lambda_ewc = lambda_ewc
        self.max_tasks  = max_tasks
        self.fisher_cap = fisher_cap
        self.device     = device

        # List of registered tasks: each is {params_star, fisher}
        self._tasks: List[Dict[str, torch.Tensor]] = []

        # Forgetting metrics: reward before/after per task
        self._forgetting_log: List[Dict] = []

        logger.info(
            f"EWC initialised | λ={lambda_ewc} | "
            f"max_tasks={max_tasks} | fisher_cap={fisher_cap}"
        )

    # ── Public API ──────────────────────────────────────────────

    def register_task(
        self,
        experience_buffer: ExperienceBuffer,
        num_samples: int = 200,
        task_name: Optional[str] = None,
    ) -> int:
        """
        Register the current policy as a completed task.

        Call this IMMEDIATELY BEFORE the agent starts adapting
        to a new attack distribution (i.e. right after drift).

        Steps:
          1. Snapshot current parameters as θ*
          2. Compute diagonal Fisher information F from experience_buffer
          3. Store (θ*, F) as a new task
          4. If max_tasks exceeded → merge oldest two tasks

        Args:
            experience_buffer: ExperienceBuffer with recent (obs, action) pairs
            num_samples:       How many samples to use for Fisher estimation
            task_name:         Optional label for logging

        Returns:
            Task index (0-indexed)
        """
        if not experience_buffer.is_ready(min_samples=50):
            logger.warning(
                f"EWC: Buffer has only {len(experience_buffer)} samples "
                f"(min 50 recommended). Fisher may be noisy."
            )

        name = task_name or f"task_{len(self._tasks)}"
        logger.info(
            f"EWC: Registering {name} | "
            f"buffer_size={len(experience_buffer)} | "
            f"fisher_samples={num_samples}"
        )

        # Step 1: Snapshot optimal parameters
        params_star = self._snapshot_params()

        # Step 2: Compute Fisher information diagonal
        fisher = self._compute_fisher(experience_buffer, num_samples)

        # Step 3: Store task
        self._tasks.append({
            "name":        name,
            "params_star": params_star,
            "fisher":      fisher,
        })

        # Step 4: Merge if over limit (online EWC)
        if len(self._tasks) > self.max_tasks:
            self._merge_oldest_tasks()

        task_idx = len(self._tasks) - 1
        logger.info(
            f"EWC: {name} registered as task #{task_idx} | "
            f"total tasks={len(self._tasks)}"
        )
        return task_idx

    def penalty(self, current_policy: Optional[nn.Module] = None) -> torch.Tensor:
        """
        Compute the total EWC penalty across all registered tasks.

        L_EWC = Σ_k (λ/2) · Σᵢ Fᵢ^k · (θᵢ - θ*ᵢ^k)²

        Called inside CVaRPPO.train() on every minibatch.

        Args:
            current_policy: Policy to compute penalty for.
                            Defaults to self.policy if None.

        Returns:
            Scalar tensor — add to PPO loss before backward()
        """
        if not self._tasks:
            return torch.tensor(0.0, device=self.device)

        model = current_policy if current_policy is not None else self.policy
        penalty = torch.tensor(0.0, device=self.device)

        for task in self._tasks:
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                if name not in task["params_star"]:
                    continue

                fisher     = task["fisher"][name].to(self.device)
                params_star = task["params_star"][name].to(self.device)

                # Quadratic anchor: Fᵢ · (θᵢ - θ*ᵢ)²
                penalty += (fisher * (param - params_star).pow(2)).sum()

        return (self.lambda_ewc / 2.0) * penalty

    def forgetting_metric(
        self, current_policy: Optional[nn.Module] = None
    ) -> float:
        """
        Scalar forgetting metric = raw EWC penalty value.

        Higher value means the current policy has drifted further
        from the consolidated task optima.
        For the paper: compare this before/after adaptation.

        Returns:
            float — 0 means perfect retention, higher = more forgetting
        """
        return self.penalty(current_policy).item()

    def per_task_forgetting(
        self, current_policy: Optional[nn.Module] = None
    ) -> Dict[str, float]:
        """
        Per-task forgetting breakdown.

        Returns dict mapping task_name → forgetting_score.
        Useful for identifying WHICH attack type the agent is forgetting.
        """
        if not self._tasks:
            return {}

        model = current_policy if current_policy is not None else self.policy
        results = {}

        for task in self._tasks:
            task_penalty = 0.0
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                if name not in task["params_star"]:
                    continue

                fisher      = task["fisher"][name].to(self.device)
                params_star = task["params_star"][name].to(self.device)
                task_penalty += (fisher * (param - params_star).pow(2)).sum().item()

            results[task["name"]] = (self.lambda_ewc / 2.0) * task_penalty

        return results

    def get_fisher_summary(self) -> Dict:
        """
        Summary statistics of Fisher matrices across all tasks.
        Useful for understanding which layers EWC is anchoring most.
        """
        if not self._tasks:
            return {}

        summary = {}
        for task in self._tasks:
            task_info = {}
            for name, fisher in task["fisher"].items():
                task_info[name] = {
                    "mean":  float(fisher.mean()),
                    "max":   float(fisher.max()),
                    "nnz":   int((fisher > 1e-6).sum()),  # non-zero count
                }
            summary[task["name"]] = task_info

        return summary

    # ── Private methods ─────────────────────────────────────────

    def _snapshot_params(self) -> Dict[str, torch.Tensor]:
        """Clone current policy parameters as θ*."""
        return {
            name: param.detach().clone()
            for name, param in self.policy.named_parameters()
            if param.requires_grad
        }

    def _compute_fisher(
        self,
        experience_buffer: ExperienceBuffer,
        num_samples: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Estimate diagonal Fisher information matrix.

        F_i = E[ (∂ log π(a|s) / ∂θᵢ)² ]

        Estimated empirically:
          F_i ≈ (1/N) Σₙ (∂ log π(aₙ|sₙ) / ∂θᵢ)²

        This is the standard online Fisher approximation used in
        Kirkpatrick et al. (2017).

        Key details:
        - We backprop through log π(a|s) to get gradients
        - We square and accumulate the gradients (not the loss gradient)
        - We cap Fisher values to prevent numerical instability
        - The policy is briefly set to eval mode (no dropout noise)
        """
        # Initialise Fisher accumulators at zero
        fisher: Dict[str, torch.Tensor] = {
            name: torch.zeros_like(param)
            for name, param in self.policy.named_parameters()
            if param.requires_grad
        }

        self.policy.eval()
        count = 0

        for obs_batch, action_batch in experience_buffer.iter_batches(
            batch_size=32, device=self.device
        ):
            if count >= num_samples:
                break

            self.policy.zero_grad()

            # Forward pass through policy
            # SB3 ActorCriticPolicy.evaluate_actions returns:
            #   (values, log_prob, entropy)
            try:
                _, log_prob, _ = self.policy.evaluate_actions(
                    obs_batch, action_batch
                )
            except Exception as e:
                # Fallback: use get_distribution for non-SB3 policies
                try:
                    dist = self.policy.get_distribution(obs_batch)
                    log_prob = dist.log_prob(action_batch)
                except Exception as e2:
                    logger.warning(f"EWC Fisher forward pass failed: {e2}")
                    continue

            # Backprop through log π(a|s)
            # We use mean log prob, then accumulate squared gradients
            loss = -log_prob.mean()
            loss.backward()

            # Accumulate squared gradients → Fisher diagonal
            for name, param in self.policy.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.detach().pow(2)

            count += obs_batch.shape[0]

        # Normalise by sample count
        n = max(count, 1)
        for name in fisher:
            fisher[name] /= n
            # Cap Fisher values to prevent instability
            fisher[name].clamp_(max=self.fisher_cap)

        self.policy.train()

        total_params = sum(f.numel() for f in fisher.values())
        nonzero = sum((f > 1e-8).sum().item() for f in fisher.values())
        logger.info(
            f"EWC Fisher computed | samples={count} | "
            f"params={total_params:,} | non-zero={nonzero:,} "
            f"({100*nonzero/max(total_params,1):.1f}%)"
        )

        return fisher

    def _merge_oldest_tasks(self):
        """
        Online EWC: merge oldest two tasks when max_tasks exceeded.

        Merged Fisher = elementwise max of both Fishers
        Merged params = average of both optimal params

        This approximation (Schwarz et al., 2018 "Progress & Compress")
        keeps memory bounded while retaining consolidated knowledge.
        """
        if len(self._tasks) < 2:
            return

        t0 = self._tasks.pop(0)
        t1 = self._tasks.pop(0)

        merged_fisher = {}
        merged_params = {}

        for name in t0["params_star"]:
            # Take max Fisher (most conservative — preserve both task memories)
            merged_fisher[name] = torch.max(
                t0["fisher"].get(name, torch.zeros(1)),
                t1["fisher"].get(name, torch.zeros(1)),
            )
            # Average optimal parameters
            merged_params[name] = (
                t0["params_star"][name] + t1["params_star"][name]
            ) / 2.0

        merged_task = {
            "name":        f"merged_{t0['name']}+{t1['name']}",
            "params_star": merged_params,
            "fisher":      merged_fisher,
        }

        # Insert merged task at front
        self._tasks.insert(0, merged_task)
        logger.info(
            f"EWC: Merged '{t0['name']}' + '{t1['name']}' "
            f"→ '{merged_task['name']}'"
        )

    # ── Properties ──────────────────────────────────────────────

    @property
    def num_tasks(self) -> int:
        return len(self._tasks)

    @property
    def task_names(self) -> List[str]:
        return [t["name"] for t in self._tasks]

    def get_status(self) -> Dict:
        return {
            "num_tasks": self.num_tasks,
            "task_names": self.task_names,
            "lambda_ewc": self.lambda_ewc,
            "forgetting_metric": self.forgetting_metric(),
            "per_task_forgetting": self.per_task_forgetting(),
        }