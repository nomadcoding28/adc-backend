"""
Continual Learner
==================
Novelty 1 — Orchestrator for the full continual learning pipeline.

Ties together:
  - DriftDetector    → detects when attack distribution has shifted
  - ExperienceBuffer → stores recent (obs, action) pairs for Fisher
  - EWC              → prevents forgetting old attack defences
  - CVaRPPO          → adapts to new distribution with EWC penalty

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THE FULL LIFECYCLE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Phase 0 — Warmup (steps 0 → warmup_steps):
  Agent trains normally on base attack distribution.
  Drift detector builds reference window.
  Experience buffer fills up with (obs, action) transitions.

Phase 1 — Drift Detection (each step):
  DriftDetector.update(obs) → returns True if drift detected.
  Drift score = Wasserstein/KL between reference and current window.

Phase 2 — Drift Response (triggered on drift):
  ① BEFORE adapting:
      EWC.register_task(experience_buffer)
        → snapshot θ*  (current optimal params)
        → compute F    (Fisher from buffer)
        → task stored in EWC memory

  ② DURING adaptation:
      CVaRPPO.train() with EWC penalty active
        → L_total = L_CVaR_PPO + λ/2 · Σ Fᵢ·(θᵢ - θ*ᵢ)²
        → Policy adapts to new distribution
        → EWC penalty prevents forgetting old task

  ③ AFTER adaptation:
      Measure forgetting: penalty on old task params
      Update reference distribution for future drift detection
      Reset experience buffer for next task

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SB3 INTEGRATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SB3's PPO.learn() runs its own internal loop — we cannot
call continual_learner.step() inside it directly.

Solution: SB3 Callback system.
  ContinualLearningCallback(BaseCallback) hooks into SB3's
  training loop at every step and every rollout end.

Usage:
    cl = ContinualLearner(agent, ewc, drift_detector, config)
    callback = cl.make_callback()
    agent.model.learn(
        total_timesteps=1_000_000,
        callback=callback,
    )
"""

import numpy as np
from typing import Optional, Callable, Dict, List
from collections import deque
from loguru import logger

from agents.ewc import ElasticWeightConsolidation, ExperienceBuffer
from drift import DriftDetector

try:
    from stable_baselines3.common.callbacks import BaseCallback
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    BaseCallback = object
    logger.warning("SB3 not available — ContinualLearningCallback disabled")


# ══════════════════════════════════════════════════════════════
#  SB3 Callback — hooks into training loop
# ══════════════════════════════════════════════════════════════

class ContinualLearningCallback(BaseCallback):
    """
    SB3 callback that drives the continual learning pipeline.

    Hooks:
      _on_step()          → feed obs/action to experience buffer
                            feed obs to drift detector
                            trigger drift response if needed
      _on_rollout_end()   → log continual learning metrics to TB
    """

    def __init__(self, continual_learner, verbose: int = 0):
        super().__init__(verbose)
        self.cl = continual_learner

    def _on_step(self) -> bool:
        """Called after every env step inside SB3's rollout collection."""
        # Grab obs and action from SB3's locals
        obs    = self.locals.get("obs_tensor",
                 self.locals.get("new_obs", None))
        action = self.locals.get("actions", None)

        if obs is not None and action is not None:
            # Handle batch dimension (SB3 uses vectorised envs)
            obs_np    = obs.cpu().numpy() if hasattr(obs, "cpu") else np.array(obs)
            action_np = action.cpu().numpy() if hasattr(action, "cpu") else np.array(action)

            # For single env: shape (1, obs_dim) → (obs_dim,)
            if obs_np.ndim == 2:
                for o, a in zip(obs_np, action_np):
                    self.cl.step(o, int(a))
            else:
                self.cl.step(obs_np, int(action_np))

        return True  # True = continue training

    def _on_rollout_end(self) -> None:
        """Called at end of each rollout — log metrics to TensorBoard."""
        status = self.cl.get_status()
        self.logger.record("continual/drift_events",      status["drift_events"])
        self.logger.record("continual/total_steps",       status["total_steps"])
        self.logger.record("continual/ewc_tasks",         status["ewc_tasks_registered"])
        self.logger.record("continual/forgetting_metric", status["forgetting_metric"])
        self.logger.record("continual/buffer_size",       status["buffer_size"])
        self.logger.record("continual/drift_score",       status["latest_drift_score"])


# ══════════════════════════════════════════════════════════════
#  ContinualLearner — main orchestrator
# ══════════════════════════════════════════════════════════════

class ContinualLearner:
    """
    Orchestrates the full Novelty 1 pipeline.

    Wires DriftDetector → ExperienceBuffer → EWC → CVaRPPO
    into a coherent continual learning loop.

    Usage:
        cl = ContinualLearner(agent, ewc, drift_detector, config)
        callback = cl.make_callback()
        agent.train(total_timesteps=1_000_000, callback=callback)
    """

    def __init__(
        self,
        agent,
        ewc: ElasticWeightConsolidation,
        drift_detector: DriftDetector,
        config: dict,
        on_drift_callback: Optional[Callable] = None,
    ):
        self.agent          = agent
        self.ewc            = ewc
        self.drift_detector = drift_detector
        self.config         = config
        self.on_drift_callback = on_drift_callback

        ewc_cfg   = config.get("ewc", {})
        drift_cfg = config.get("drift", {})

        # Config values
        self._warmup_steps    = drift_cfg.get("warmup_steps", 5000)
        self._alert_cooldown  = drift_cfg.get("alert_cooldown", 2000)
        self._fisher_samples  = ewc_cfg.get("fisher_samples", 200)
        self._adapt_steps     = ewc_cfg.get("fisher_samples", 200) * 20
        self._buffer_capacity = ewc_cfg.get("buffer_capacity", 2000)

        # Experience buffer — fills up during normal training
        self.experience_buffer = ExperienceBuffer(
            capacity=self._buffer_capacity
        )

        # State
        self._steps           = 0
        self._drift_events    = 0
        self._last_drift_step = 0
        self._is_adapting     = False

        # History for metrics / plotting
        self._drift_history:       List[Dict] = []
        self._forgetting_history:  List[float] = []
        self._adaptation_speeds:   List[int] = []   # steps to recover reward

        logger.info(
            f"ContinualLearner initialised | "
            f"warmup={self._warmup_steps} | "
            f"cooldown={self._alert_cooldown} | "
            f"adapt_steps={self._adapt_steps} | "
            f"buffer={self._buffer_capacity}"
        )

    # ── Core step — call every env step ────────────────────────

    def step(self, observation: np.ndarray, action: int):
        """
        Process one environment transition.

        This is called by ContinualLearningCallback._on_step()
        inside SB3's training loop.

        Args:
            observation: Current obs (after action was taken)
            action:      Action taken this step
        """
        self._steps += 1

        # 1. Store in experience buffer (for Fisher computation)
        self.experience_buffer.add(observation, action)

        # 2. Feed to drift detector
        drift_detected = self.drift_detector.update(observation)

        # 3. Trigger adaptation if drift detected and cooldown passed
        if (
            drift_detected
            and not self._is_adapting
            and self._can_trigger()
        ):
            self._handle_drift()

    # ── Drift response ──────────────────────────────────────────

    def _can_trigger(self) -> bool:
        """Check warmup and cooldown conditions."""
        warmed_up   = self._steps >= self._warmup_steps
        cooled_down = (
            self._steps - self._last_drift_step
        ) >= self._alert_cooldown
        return warmed_up and cooled_down

    def _handle_drift(self):
        """
        Full drift response pipeline.
        This is the heart of Novelty 1.
        """
        self._is_adapting     = True
        self._drift_events   += 1
        self._last_drift_step = self._steps
        drift_score           = self.drift_detector.get_drift_score()

        logger.warning(
            f"\n{'━'*55}\n"
            f"  ⚠️  DRIFT DETECTED — event #{self._drift_events}\n"
            f"  Step:        {self._steps:,}\n"
            f"  Drift score: {drift_score:.4f}\n"
            f"  Buffer size: {len(self.experience_buffer)}\n"
            f"{'━'*55}"
        )

        # ── Step 1: Register current task with EWC ─────────────
        # This MUST happen before any parameter updates.
        # It snapshots the current policy as the "optimal" params
        # for the old attack distribution.
        if self.ewc is not None and self.experience_buffer.is_ready(50):
            pre_forgetting = self.ewc.forgetting_metric()

            task_idx = self.ewc.register_task(
                experience_buffer=self.experience_buffer,
                num_samples=self._fisher_samples,
                task_name=f"drift_{self._drift_events}_step{self._steps}",
            )

            logger.info(
                f"EWC task #{task_idx} registered | "
                f"total_tasks={self.ewc.num_tasks}"
            )
        else:
            if self.ewc is None:
                logger.warning("EWC not attached — forgetting NOT prevented")
            elif not self.experience_buffer.is_ready(50):
                logger.warning(
                    f"Buffer too small ({len(self.experience_buffer)} samples) "
                    f"— EWC skipped for this drift event"
                )
            pre_forgetting = 0.0

        # ── Step 2: Attach EWC to CVaRPPO (if not already) ─────
        # EWC penalty will now be active in CVaRPPO.train()
        if self.ewc is not None and hasattr(self.agent, "attach_ewc"):
            self.agent.attach_ewc(self.ewc)

        # ── Step 3: Incremental adaptation ─────────────────────
        # Train on NEW distribution WITH EWC penalty active.
        # reset_num_timesteps=False preserves the step counter
        # so SB3 knows training is continuing, not restarting.
        logger.info(
            f"Starting incremental adaptation | "
            f"steps={self._adapt_steps:,}"
        )
        self.agent.train(
            total_timesteps=self._adapt_steps,
            reset_num_timesteps=False,
        )

        # ── Step 4: Measure forgetting after adaptation ─────────
        post_forgetting = self.ewc.forgetting_metric() if self.ewc else 0.0
        forgetting_delta = post_forgetting - pre_forgetting
        self._forgetting_history.append(post_forgetting)

        # ── Step 5: Reset experience buffer for next task ───────
        # The buffer should now collect samples from the NEW distribution
        self.experience_buffer.clear()

        # ── Step 6: Log drift event ─────────────────────────────
        drift_event = {
            "event":           self._drift_events,
            "step":            self._steps,
            "drift_score":     drift_score,
            "ewc_tasks_after": self.ewc.num_tasks if self.ewc else 0,
            "forgetting_pre":  pre_forgetting,
            "forgetting_post": post_forgetting,
            "forgetting_delta": forgetting_delta,
            "adapt_steps":     self._adapt_steps,
        }
        self._drift_history.append(drift_event)

        logger.info(
            f"Drift response complete | "
            f"forgetting={post_forgetting:.4f} "
            f"(Δ={forgetting_delta:+.4f})"
        )

        # ── Step 7: External callback (e.g. dashboard alert) ────
        if self.on_drift_callback:
            self.on_drift_callback(**drift_event)

        self._is_adapting = False

    # ── SB3 callback factory ────────────────────────────────────

    def make_callback(self) -> "ContinualLearningCallback":
        """
        Create an SB3 callback that drives this ContinualLearner.

        Pass the returned callback to agent.model.learn():
            callback = cl.make_callback()
            agent.model.learn(1_000_000, callback=callback)
        """
        if not SB3_AVAILABLE:
            logger.warning("SB3 not available — returning None callback")
            return None
        return ContinualLearningCallback(self)

    # ── Manual feed (without SB3 callback) ─────────────────────

    def feed_rollout_buffer(self, rollout_buffer):
        """
        Manually populate the experience buffer from an SB3 rollout buffer.

        Call this at the end of each rollout if NOT using the callback:
            cl.feed_rollout_buffer(agent.model.rollout_buffer)
        """
        try:
            obs_arr    = rollout_buffer.observations
            action_arr = rollout_buffer.actions

            if obs_arr.ndim == 3:   # (n_steps, n_envs, obs_dim)
                obs_arr    = obs_arr.reshape(-1, obs_arr.shape[-1])
                action_arr = action_arr.reshape(-1)

            self.experience_buffer.add_batch(obs_arr, action_arr)

        except Exception as e:
            logger.warning(f"feed_rollout_buffer failed: {e}")

    # ── Status & metrics ────────────────────────────────────────

    def get_status(self) -> Dict:
        return {
            "total_steps":         self._steps,
            "drift_events":        self._drift_events,
            "last_drift_step":     self._last_drift_step,
            "ewc_tasks_registered": self.ewc.num_tasks if self.ewc else 0,
            "forgetting_metric":   self.ewc.forgetting_metric() if self.ewc else 0.0,
            "buffer_size":         len(self.experience_buffer),
            "is_adapting":         self._is_adapting,
            "latest_drift_score":  self.drift_detector.get_drift_score(),
            "drift_detector":      self.drift_detector.get_status(),
        }

    def get_drift_history(self) -> List[Dict]:
        """Full history of drift events — use for paper plots."""
        return list(self._drift_history)

    def get_forgetting_curve(self) -> List[float]:
        """
        Forgetting metric after each drift event.
        Should stay low with EWC, spike without it.
        Key figure for the paper.
        """
        return list(self._forgetting_history)

    def get_adaptation_summary(self) -> Dict:
        """Summary statistics for the paper's results table."""
        if not self._drift_history:
            return {"drift_events": 0}

        forgetting_deltas = [e["forgetting_delta"] for e in self._drift_history]
        drift_scores      = [e["drift_score"] for e in self._drift_history]

        return {
            "total_drift_events":       self._drift_events,
            "mean_drift_score":         float(np.mean(drift_scores)),
            "mean_forgetting_delta":    float(np.mean(forgetting_deltas)),
            "max_forgetting":           float(max(self._forgetting_history or [0])),
            "ewc_tasks_accumulated":    self.ewc.num_tasks if self.ewc else 0,
            "drift_events":             self._drift_history,
        }

    @property
    def drift_events(self) -> int:
        return self._drift_events

    @property
    def total_steps(self) -> int:
        return self._steps