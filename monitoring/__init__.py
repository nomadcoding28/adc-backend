"""
monitoring/
===========
Observability stack for the ACD Framework.

Covers three pillars of production observability:

    Metrics  → Prometheus (prometheus.py)
               Custom gauges, counters, and histograms for training,
               risk metrics, drift detection, and WebSocket connections.
               Scraped by Prometheus every 15s; visualised in Grafana.

    Errors   → Sentry (sentry.py)
               Captures unhandled exceptions from FastAPI, Celery tasks,
               and the training loop. Integrates with FastAPI + Celery +
               SQLAlchemy automatically.

    Logs     → Structured logging (structlog_config.py)
               JSON lines in production (parseable by Datadog, CloudWatch),
               coloured human-readable output in development.

Quick-start
-----------
    from monitoring import configure_logging, PrometheusMetrics
    from monitoring.sentry import init_sentry

    # Call once at application startup (in main.py)
    configure_logging()
    init_sentry()

    # Use metrics anywhere in the codebase
    metrics = PrometheusMetrics()
    metrics.update_reward(8.74)
    metrics.increment_drift_events()
"""

from monitoring.prometheus import PrometheusMetrics
from monitoring.structlog_config import configure_logging
from monitoring.sentry import init_sentry, capture_exception, capture_message

__all__ = [
    "PrometheusMetrics",
    "configure_logging",
    "init_sentry",
    "capture_exception",
    "capture_message",
]