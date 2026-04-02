"""
monitoring/prometheus.py
=========================
Custom Prometheus metrics for the ACD Framework.

Exposes training, risk, drift, game-model, and infrastructure metrics
for scraping by Prometheus and visualisation in Grafana.

Metric categories
-----------------
    Training     : mean_reward, episode_reward histogram, timesteps,
                   episodes, policy/value/EWC loss, training active flag
    Risk (CVaR)  : cvar_005, cvar_001, catastrophic_rate, success_rate
    Drift        : drift_score, drift_events_total, ewc_forgetting,
                   ewc_tasks_registered
    Game model   : nash_game_value, belief_entropy, attacker_prediction_accuracy
    Infrastructure: ws_connections, api_request_duration, llm_latency,
                    api_request_total, kg_stats

Prometheus data model
---------------------
Each Prometheus metric type serves a different purpose:
    Gauge    — current value that can go up or down (reward, CVaR, drift score)
    Counter  — monotonically increasing (timesteps, events, request count)
    Histogram— distribution of values (latencies, episode rewards)

Usage
-----
    # Get the singleton instance
    metrics = PrometheusMetrics()

    # Update from training loop
    metrics.update_reward(8.74)
    metrics.update_cvar(alpha=0.05, value=-2.14)
    metrics.increment_timesteps(2048)
    metrics.increment_drift_events()

    # Update all agent metrics at once
    metrics.update_from_agent_metrics(agent.get_metrics())

    # Record API request (called from middleware)
    metrics.record_api_request("GET", "/network/topology", 200, 0.012)

Grafana dashboard tip
---------------------
Import the ACD Framework dashboard from grafana/acd_dashboard.json
(included in the repository). Panels are pre-configured for all
metrics defined here.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Optional prometheus_client — degrade gracefully if not installed
try:
    from prometheus_client import (
        Counter,
        Gauge,
        Histogram,
        Summary,
        CollectorRegistry,
        REGISTRY,
    )
    _PROM_AVAILABLE = True
except ImportError:
    _PROM_AVAILABLE = False
    logger.warning(
        "prometheus_client not installed — Prometheus metrics disabled. "
        "Install with: pip install prometheus-client"
    )


# ── Histogram bucket definitions ────────────────────────────────────────────

# Episode reward buckets (CybORG Scenario2 range)
_REWARD_BUCKETS = [-50.0, -20.0, -10.0, -5.0, -2.0, 0.0, 2.0, 5.0, 8.0, 10.0, 15.0, 20.0, 50.0]

# API latency buckets (seconds)
_LATENCY_BUCKETS = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0]

# LLM / RAG latency buckets (seconds — slower than API)
_LLM_LATENCY_BUCKETS = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0]

# Drift score buckets (Wasserstein distance)
_DRIFT_BUCKETS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0]


class PrometheusMetrics:
    """
    Singleton wrapper around prometheus_client metric objects.

    All metric objects are registered once at first instantiation.
    Subsequent calls to ``PrometheusMetrics()`` return the same instance
    and the same underlying metric objects.

    Thread-safe — prometheus_client metrics are designed for concurrent use.

    Parameters
    ----------
    namespace : str
        Metric name prefix.  Default ``"acd"``.
        All metric names will be ``{namespace}_{metric_name}``.

    Example metric names with default namespace
    -------------------------------------------
        acd_mean_reward
        acd_cvar_005
        acd_timesteps_total
        acd_drift_events_total
        acd_api_request_duration_seconds
    """

    _instance: Optional["PrometheusMetrics"] = None
    _initialised: bool = False

    def __new__(cls, namespace: str = "acd") -> "PrometheusMetrics":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, namespace: str = "acd") -> None:
        if self._initialised:
            return
        self._initialised = True
        self._namespace    = namespace
        self._prefix       = f"{namespace}_"

        if not _PROM_AVAILABLE:
            logger.warning(
                "PrometheusMetrics: prometheus_client not installed — "
                "all update calls will be no-ops."
            )
            return

        self._register_metrics()
        logger.info("Prometheus metrics registered (namespace=%r).", namespace)

    # ================================================================== #
    # Metric registration
    # ================================================================== #

    def _register_metrics(self) -> None:
        """Create and register all Prometheus metric objects."""

        p = self._prefix   # shorthand

        # ── Training metrics ───────────────────────────────────────────

        self.mean_reward = Gauge(
            f"{p}mean_reward",
            "Mean episode reward over the last 100 episodes.",
        )
        self.is_training = Gauge(
            f"{p}is_training",
            "1 while a training run is active, 0 otherwise.",
        )
        self.timesteps_total = Counter(
            f"{p}timesteps_total",
            "Total environment interaction steps across all training runs.",
        )
        self.episodes_total = Counter(
            f"{p}episodes_total",
            "Total training episodes completed.",
        )
        self.loss_policy = Gauge(
            f"{p}loss_policy",
            "Most recent policy gradient loss value.",
        )
        self.loss_value = Gauge(
            f"{p}loss_value",
            "Most recent value function loss.",
        )
        self.loss_ewc = Gauge(
            f"{p}loss_ewc",
            "Most recent EWC regularisation penalty loss.",
        )
        self.explained_variance = Gauge(
            f"{p}explained_variance",
            "Value function explained variance (quality indicator, 0–1).",
        )
        self.episode_reward_histogram = Histogram(
            f"{p}episode_reward",
            "Distribution of episode total rewards.",
            buckets=_REWARD_BUCKETS,
        )

        # ── CVaR / risk metrics ────────────────────────────────────────

        self.cvar_001 = Gauge(
            f"{p}cvar_001",
            "Conditional Value-at-Risk at α=0.01 (worst 1% episodes).",
        )
        self.cvar_005 = Gauge(
            f"{p}cvar_005",
            "Conditional Value-at-Risk at α=0.05 (worst 5% episodes) — primary paper metric.",
        )
        self.cvar_010 = Gauge(
            f"{p}cvar_010",
            "Conditional Value-at-Risk at α=0.10 (worst 10% episodes).",
        )
        self.cvar_050 = Gauge(
            f"{p}cvar_050",
            "Conditional Value-at-Risk at α=0.50 (near risk-neutral baseline).",
        )
        self.catastrophic_rate = Gauge(
            f"{p}catastrophic_rate",
            "Fraction of episodes with catastrophic outcome (reward < -10).",
        )
        self.success_rate = Gauge(
            f"{p}success_rate",
            "Fraction of episodes with positive reward (successful defence).",
        )
        self.var_005 = Gauge(
            f"{p}var_005",
            "Value-at-Risk at α=0.05 (5th percentile reward threshold).",
        )

        # ── EWC / continual learning metrics ──────────────────────────

        self.ewc_forgetting = Gauge(
            f"{p}ewc_forgetting",
            "EWC forgetting metric — lower is better (0 = no forgetting).",
        )
        self.ewc_tasks_registered = Gauge(
            f"{p}ewc_tasks_registered",
            "Number of EWC tasks (past attack distributions) registered.",
        )
        self.task_retention = Gauge(
            f"{p}task_retention",
            "Mean task retention rate across all registered EWC tasks.",
        )

        # ── Drift detection metrics ────────────────────────────────────

        self.drift_score = Gauge(
            f"{p}drift_score",
            "Current Wasserstein-1 distributional distance (drift detector primary metric).",
        )
        self.drift_threshold = Gauge(
            f"{p}drift_threshold",
            "Configured drift detection threshold.",
        )
        self.drift_events_total = Counter(
            f"{p}drift_events_total",
            "Total concept drift events detected since application start.",
        )
        self.drift_cooldown_remaining = Gauge(
            f"{p}drift_cooldown_remaining",
            "Steps remaining before the next drift event can be triggered.",
        )
        self.drift_score_histogram = Histogram(
            f"{p}drift_score_distribution",
            "Distribution of observed drift scores over time.",
            buckets=_DRIFT_BUCKETS,
        )

        # ── Game model / Bayesian belief metrics ──────────────────────

        self.belief_entropy = Gauge(
            f"{p}belief_entropy_bits",
            "Shannon entropy of the attacker type belief distribution (bits, max=1.58).",
        )
        self.dominant_attacker_probability = Gauge(
            f"{p}dominant_attacker_probability",
            "Posterior probability of the most likely attacker type.",
        )
        self.nash_game_value = Gauge(
            f"{p}nash_game_value",
            "Nash equilibrium game value V* from the most recent LP solve.",
        )
        self.nash_exploitability = Gauge(
            f"{p}nash_exploitability",
            "Nash exploitability — distance from true Nash equilibrium (lower = better).",
        )
        self.attacker_prediction_accuracy = Gauge(
            f"{p}attacker_prediction_accuracy",
            "Fraction of steps where the predicted attacker action matched actual.",
        )
        self.belief_shift_count = Counter(
            f"{p}belief_shift_total",
            "Total number of times the dominant attacker type changed.",
        )

        # ── Network / environment metrics ──────────────────────────────

        self.hosts_compromised = Gauge(
            f"{p}hosts_compromised",
            "Number of currently compromised hosts in the simulated network.",
        )
        self.hosts_isolated = Gauge(
            f"{p}hosts_isolated",
            "Number of currently isolated hosts.",
        )
        self.decoys_active = Gauge(
            f"{p}decoys_active",
            "Number of active decoy hosts.",
        )
        self.episode_step = Gauge(
            f"{p}episode_step",
            "Current episode step number.",
        )

        # ── API / infrastructure metrics ──────────────────────────────

        self.ws_connections = Gauge(
            f"{p}websocket_connections",
            "Number of currently active WebSocket connections.",
        )
        self.api_requests_total = Counter(
            f"{p}api_requests_total",
            "Total HTTP API requests handled.",
            labelnames=["method", "endpoint", "status_code"],
        )
        self.api_request_duration = Histogram(
            f"{p}api_request_duration_seconds",
            "HTTP API request latency in seconds.",
            labelnames=["method", "endpoint"],
            buckets=_LATENCY_BUCKETS,
        )
        self.llm_request_duration = Histogram(
            f"{p}llm_request_duration_seconds",
            "LLM explanation request latency in seconds.",
            labelnames=["provider", "model"],
            buckets=_LLM_LATENCY_BUCKETS,
        )
        self.llm_tokens_used = Counter(
            f"{p}llm_tokens_total",
            "Total LLM tokens consumed (prompt + completion).",
            labelnames=["provider"],
        )
        self.rag_retrieval_duration = Histogram(
            f"{p}rag_retrieval_duration_seconds",
            "RAG document retrieval latency in seconds.",
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
        )
        self.celery_tasks_total = Counter(
            f"{p}celery_tasks_total",
            "Total Celery background tasks submitted.",
            labelnames=["task_name", "status"],
        )

        # ── KG metrics ─────────────────────────────────────────────────

        self.kg_cve_count = Gauge(
            f"{p}kg_cves_indexed",
            "Total CVE nodes in the knowledge graph.",
        )
        self.kg_technique_count = Gauge(
            f"{p}kg_techniques_indexed",
            "Total ATT&CK technique nodes in the knowledge graph.",
        )
        self.kg_edge_count = Gauge(
            f"{p}kg_edges_total",
            "Total edges (relationships) in the knowledge graph.",
        )
        self.kg_last_built = Gauge(
            f"{p}kg_last_built_timestamp",
            "Unix timestamp of the last full KG rebuild.",
        )

    # ================================================================== #
    # Update helpers — Training
    # ================================================================== #

    def update_reward(self, mean_reward: float) -> None:
        """Update mean reward gauge and record in the episode reward histogram."""
        if not _PROM_AVAILABLE:
            return
        try:
            self.mean_reward.set(mean_reward)
            self.episode_reward_histogram.observe(mean_reward)
        except Exception as exc:
            logger.debug("Prometheus update_reward failed: %s", exc)

    def update_losses(
        self,
        policy:   Optional[float] = None,
        value:    Optional[float] = None,
        ewc:      Optional[float] = None,
        explained_variance: Optional[float] = None,
    ) -> None:
        """Update training loss gauges."""
        if not _PROM_AVAILABLE:
            return
        try:
            if policy   is not None: self.loss_policy.set(policy)
            if value    is not None: self.loss_value.set(value)
            if ewc      is not None: self.loss_ewc.set(ewc)
            if explained_variance is not None:
                self.explained_variance.set(explained_variance)
        except Exception as exc:
            logger.debug("Prometheus update_losses failed: %s", exc)

    def set_training_active(self, active: bool) -> None:
        """Set the is_training flag (1 = training, 0 = stopped)."""
        if not _PROM_AVAILABLE:
            return
        try:
            self.is_training.set(1.0 if active else 0.0)
        except Exception:
            pass

    def increment_timesteps(self, n: int = 1) -> None:
        """Increment the total timesteps counter by n."""
        if not _PROM_AVAILABLE:
            return
        try:
            self.timesteps_total.inc(n)
        except Exception:
            pass

    def increment_episodes(self, n: int = 1) -> None:
        """Increment the total episodes counter by n."""
        if not _PROM_AVAILABLE:
            return
        try:
            self.episodes_total.inc(n)
        except Exception:
            pass

    def observe_episode_reward(self, reward: float) -> None:
        """Record a single episode reward in the histogram."""
        if not _PROM_AVAILABLE:
            return
        try:
            self.episode_reward_histogram.observe(reward)
        except Exception:
            pass

    # ================================================================== #
    # Update helpers — CVaR / Risk
    # ================================================================== #

    def update_cvar(self, alpha: float, value: float) -> None:
        """
        Update the CVaR gauge for a specific α level.

        Parameters
        ----------
        alpha : float
            Risk level (0.01, 0.05, 0.10, 0.50).
        value : float
            CVaR value (expected reward in worst α-fraction of episodes).
        """
        if not _PROM_AVAILABLE:
            return
        try:
            if   abs(alpha - 0.01) < 1e-4: self.cvar_001.set(value)
            elif abs(alpha - 0.05) < 1e-4: self.cvar_005.set(value)
            elif abs(alpha - 0.10) < 1e-4: self.cvar_010.set(value)
            elif abs(alpha - 0.50) < 1e-4: self.cvar_050.set(value)
        except Exception as exc:
            logger.debug("Prometheus update_cvar failed: %s", exc)

    def update_risk_metrics(
        self,
        cvar_005:          Optional[float] = None,
        cvar_001:          Optional[float] = None,
        cvar_010:          Optional[float] = None,
        var_005:           Optional[float] = None,
        catastrophic_rate: Optional[float] = None,
        success_rate:      Optional[float] = None,
    ) -> None:
        """Bulk update all CVaR / risk metrics."""
        if not _PROM_AVAILABLE:
            return
        try:
            if cvar_005          is not None: self.cvar_005.set(cvar_005)
            if cvar_001          is not None: self.cvar_001.set(cvar_001)
            if cvar_010          is not None: self.cvar_010.set(cvar_010)
            if var_005           is not None: self.var_005.set(var_005)
            if catastrophic_rate is not None: self.catastrophic_rate.set(catastrophic_rate)
            if success_rate      is not None: self.success_rate.set(success_rate)
        except Exception as exc:
            logger.debug("Prometheus update_risk_metrics failed: %s", exc)

    # ================================================================== #
    # Update helpers — EWC / Continual Learning
    # ================================================================== #

    def update_ewc_forgetting(self, forgetting: float) -> None:
        """Update the EWC forgetting metric gauge."""
        if not _PROM_AVAILABLE:
            return
        try:
            self.ewc_forgetting.set(forgetting)
        except Exception:
            pass

    def update_ewc_tasks(self, n_tasks: int, retention: Optional[float] = None) -> None:
        """Update EWC task count and optional mean retention rate."""
        if not _PROM_AVAILABLE:
            return
        try:
            self.ewc_tasks_registered.set(n_tasks)
            if retention is not None:
                self.task_retention.set(retention)
        except Exception:
            pass

    # ================================================================== #
    # Update helpers — Drift Detection
    # ================================================================== #

    def update_drift_score(self, score: float) -> None:
        """Update the current drift distance score."""
        if not _PROM_AVAILABLE:
            return
        try:
            self.drift_score.set(score)
            self.drift_score_histogram.observe(score)
        except Exception:
            pass

    def update_drift_threshold(self, threshold: float) -> None:
        """Update the configured drift threshold gauge."""
        if not _PROM_AVAILABLE:
            return
        try:
            self.drift_threshold.set(threshold)
        except Exception:
            pass

    def increment_drift_events(self, n: int = 1) -> None:
        """Increment the drift events counter."""
        if not _PROM_AVAILABLE:
            return
        try:
            self.drift_events_total.inc(n)
        except Exception:
            pass

    def update_drift_cooldown(self, remaining_steps: int) -> None:
        """Update the drift cooldown remaining steps gauge."""
        if not _PROM_AVAILABLE:
            return
        try:
            self.drift_cooldown_remaining.set(remaining_steps)
        except Exception:
            pass

    # ================================================================== #
    # Update helpers — Game Model
    # ================================================================== #

    def update_belief(
        self,
        entropy:               float,
        dominant_probability:  float,
    ) -> None:
        """Update Bayesian belief metrics."""
        if not _PROM_AVAILABLE:
            return
        try:
            self.belief_entropy.set(entropy)
            self.dominant_attacker_probability.set(dominant_probability)
        except Exception:
            pass

    def update_nash(self, game_value: float, exploitability: float) -> None:
        """Update Nash equilibrium solver metrics."""
        if not _PROM_AVAILABLE:
            return
        try:
            self.nash_game_value.set(game_value)
            self.nash_exploitability.set(exploitability)
        except Exception:
            pass

    def update_attacker_prediction(self, accuracy: float) -> None:
        """Update attacker next-action prediction accuracy."""
        if not _PROM_AVAILABLE:
            return
        try:
            self.attacker_prediction_accuracy.set(accuracy)
        except Exception:
            pass

    def increment_belief_shifts(self) -> None:
        """Increment the belief shift counter (attacker type changed)."""
        if not _PROM_AVAILABLE:
            return
        try:
            self.belief_shift_count.inc()
        except Exception:
            pass

    # ================================================================== #
    # Update helpers — Network / Environment
    # ================================================================== #

    def update_network_state(
        self,
        n_compromised: int,
        n_isolated:    int = 0,
        n_decoys:      int = 0,
        episode_step:  int = 0,
    ) -> None:
        """Update network host state gauges."""
        if not _PROM_AVAILABLE:
            return
        try:
            self.hosts_compromised.set(n_compromised)
            self.hosts_isolated.set(n_isolated)
            self.decoys_active.set(n_decoys)
            self.episode_step.set(episode_step)
        except Exception:
            pass

    # ================================================================== #
    # Update helpers — API / Infrastructure
    # ================================================================== #

    def update_ws_connections(self, n: int) -> None:
        """Update active WebSocket connection count."""
        if not _PROM_AVAILABLE:
            return
        try:
            self.ws_connections.set(n)
        except Exception:
            pass

    def record_api_request(
        self,
        method:     str,
        endpoint:   str,
        status_code:int,
        latency_s:  float,
    ) -> None:
        """
        Record a single HTTP API request.

        Parameters
        ----------
        method : str
            HTTP method (GET, POST, etc.).
        endpoint : str
            Route path (e.g. "/network/topology").
        status_code : int
            HTTP response status code.
        latency_s : float
            Request processing time in seconds.
        """
        if not _PROM_AVAILABLE:
            return
        try:
            self.api_requests_total.labels(
                method      = method,
                endpoint    = endpoint,
                status_code = str(status_code),
            ).inc()
            self.api_request_duration.labels(
                method   = method,
                endpoint = endpoint,
            ).observe(latency_s)
        except Exception as exc:
            logger.debug("Prometheus record_api_request failed: %s", exc)

    def record_llm_request(
        self,
        latency_s:  float,
        tokens:     int   = 0,
        provider:   str   = "openai",
        model:      str   = "gpt-4o-mini",
    ) -> None:
        """
        Record a single LLM explanation request.

        Parameters
        ----------
        latency_s : float
            Wall-clock time for the LLM API call.
        tokens : int
            Total tokens consumed (prompt + completion).
        provider : str
            LLM provider name (e.g. "openai", "anthropic").
        model : str
            Model name string.
        """
        if not _PROM_AVAILABLE:
            return
        try:
            self.llm_request_duration.labels(
                provider = provider,
                model    = model,
            ).observe(latency_s)
            if tokens > 0:
                self.llm_tokens_used.labels(provider=provider).inc(tokens)
        except Exception as exc:
            logger.debug("Prometheus record_llm_request failed: %s", exc)

    def record_rag_retrieval(self, latency_s: float) -> None:
        """Record a RAG vector search latency."""
        if not _PROM_AVAILABLE:
            return
        try:
            self.rag_retrieval_duration.observe(latency_s)
        except Exception:
            pass

    def record_celery_task(self, task_name: str, status: str = "submitted") -> None:
        """Record a Celery task submission or completion."""
        if not _PROM_AVAILABLE:
            return
        try:
            self.celery_tasks_total.labels(
                task_name = task_name,
                status    = status,
            ).inc()
        except Exception:
            pass

    # ================================================================== #
    # Update helpers — Knowledge Graph
    # ================================================================== #

    def update_kg_stats(
        self,
        n_cves:       int = 0,
        n_techniques: int = 0,
        n_edges:      int = 0,
    ) -> None:
        """Update knowledge graph node/edge count gauges."""
        if not _PROM_AVAILABLE:
            return
        try:
            self.kg_cve_count.set(n_cves)
            self.kg_technique_count.set(n_techniques)
            self.kg_edge_count.set(n_edges)
            self.kg_last_built.set(time.time())
        except Exception as exc:
            logger.debug("Prometheus update_kg_stats failed: %s", exc)

    # ================================================================== #
    # Bulk update from module metric dicts
    # ================================================================== #

    def update_from_agent_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Bulk update all agent-related metrics from a single dict.

        Accepts the dict returned by ``agent.get_metrics()``.
        Gracefully ignores unknown or None-valued keys.

        Parameters
        ----------
        metrics : dict
            Output of ``ACDPPOAgent.get_metrics()``.
        """
        if not metrics:
            return

        if metrics.get("mean_reward")      is not None: self.update_reward(metrics["mean_reward"])
        if metrics.get("cvar_005")         is not None: self.update_cvar(0.05, metrics["cvar_005"])
        if metrics.get("cvar_001")         is not None: self.update_cvar(0.01, metrics["cvar_001"])
        if metrics.get("cvar_010")         is not None: self.update_cvar(0.10, metrics["cvar_010"])
        if metrics.get("catastrophic_rate")is not None: self.catastrophic_rate.set(metrics["catastrophic_rate"]) if _PROM_AVAILABLE else None
        if metrics.get("success_rate")     is not None: self.success_rate.set(metrics["success_rate"]) if _PROM_AVAILABLE else None
        if metrics.get("loss_policy")      is not None: self.update_losses(policy=metrics["loss_policy"])
        if metrics.get("loss_value")       is not None: self.update_losses(value=metrics["loss_value"])
        if metrics.get("loss_ewc")         is not None: self.update_losses(ewc=metrics["loss_ewc"])
        if metrics.get("ewc_forgetting")   is not None: self.update_ewc_forgetting(metrics["ewc_forgetting"])
        if metrics.get("is_training")      is not None: self.set_training_active(metrics["is_training"])

    def update_from_drift_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Bulk update drift metrics from drift detector's ``get_metrics()`` dict.

        Parameters
        ----------
        metrics : dict
            Output of ``DriftDetector.get_metrics()``.
        """
        if not metrics:
            return

        if metrics.get("current_distance") is not None:
            self.update_drift_score(metrics["current_distance"])
        if metrics.get("threshold") is not None:
            self.update_drift_threshold(metrics["threshold"])

    def update_from_game_snapshot(self, snapshot: Dict[str, Any]) -> None:
        """
        Bulk update game model metrics from a ``GameMetricsSnapshot`` dict.

        Parameters
        ----------
        snapshot : dict
            Output of ``GameMetrics.get_snapshot().to_dict()``.
        """
        if not snapshot:
            return

        if snapshot.get("belief_entropy")               is not None:
            self.update_belief(
                entropy              = snapshot["belief_entropy"],
                dominant_probability = snapshot.get("dominant_probability", 0.0),
            )
        if snapshot.get("game_value")                   is not None:
            self.update_nash(
                game_value      = snapshot["game_value"],
                exploitability  = snapshot.get("exploitability", 0.0),
            )
        if snapshot.get("attacker_prediction_accuracy") is not None:
            self.update_attacker_prediction(snapshot["attacker_prediction_accuracy"])

    # ================================================================== #
    # Introspection
    # ================================================================== #

    @property
    def is_available(self) -> bool:
        """True if prometheus_client is installed and metrics are active."""
        return _PROM_AVAILABLE

    @property
    def namespace(self) -> str:
        """The metric namespace prefix (e.g. "acd")."""
        return self._namespace

    def list_metric_names(self) -> List[str]:
        """Return a sorted list of all registered metric names."""
        if not _PROM_AVAILABLE:
            return []
        return sorted(
            name
            for name in dir(self)
            if not name.startswith("_")
            and hasattr(getattr(self, name, None), "_name")
        )

    def __repr__(self) -> str:
        return (
            f"PrometheusMetrics("
            f"namespace={self._namespace!r}, "
            f"available={_PROM_AVAILABLE})"
        )