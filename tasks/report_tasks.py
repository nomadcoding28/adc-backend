"""
tasks/report_tasks.py
======================
Celery tasks for LLM-powered incident report generation.

Tasks
-----
    generate_report(context)
        Generate a structured incident report using the ReAct agent + LLM.

    generate_training_summary(run_id, metrics)
        Generate a post-training summary report with key metrics.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from celery import Task
from celery.exceptions import SoftTimeLimitExceeded

from tasks.celery_app import celery_app

logger = logging.getLogger(__name__)


class _BaseReportTask(Task):
    """Base task with cleanup of LLM client resources."""

    abstract = True

    def __init__(self) -> None:
        super().__init__()
        self._llm_client = None

    def after_return(self, status, retval, task_id, args, kwargs, einfo) -> None:
        self._llm_client = None

    def _get_llm_client(self, config: Dict[str, Any]) -> Any:
        """Lazily create an LLM client for this task."""
        if self._llm_client is None:
            from explainability.llm.client import LLMClient
            self._llm_client = LLMClient.from_config(config.get("llm", {}))
        return self._llm_client


# ══════════════════════════════════════════════════════════════════════════════
# Incident report generation
# ══════════════════════════════════════════════════════════════════════════════

@celery_app.task(
    bind          = True,
    base          = _BaseReportTask,
    name          = "tasks.report_tasks.generate_report",
    queue         = "reports",
    max_retries   = 1,
    track_started = True,
    ignore_result = False,
)
def generate_report(
    self:    Task,
    context: Dict[str, Any],
    config:  Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Generate a structured incident report.

    Uses the ReportGenerator (LLM + RAG + KG) to produce a full
    incident report with threat analysis, MITRE mapping, and
    recommended actions.

    Parameters
    ----------
    context : dict
        Incident context.  Expected keys:
            threat_description  : str
            affected_hosts      : list[str]
            technique_ids       : list[str], optional
            cve_ids             : list[str], optional
            severity            : str, optional
            actions_taken       : list[str], optional
    config : dict, optional
        Application configuration.

    Returns
    -------
    dict
        Full Incident report dict (matches ``api/schemas/incidents.Incident``).
    """
    start_time = time.monotonic()
    cfg     = config or _load_config()
    run_id  = (self.request.id or "local")[:8]

    logger.info("Report generation START — run_id=%s", run_id)

    self.update_state(
        state = "STARTED",
        meta  = {"run_id": run_id, "message": "Initialising LLM client..."},
    )

    try:
        llm_client = self._get_llm_client(cfg)

        from explainability.report_generator import ReportGenerator
        generator = ReportGenerator(llm=llm_client, auto_save=True)

        self.update_state(
            state = "PROGRESS",
            meta  = {"run_id": run_id, "message": "Generating report via LLM..."},
        )

        report  = generator.generate(context)
        elapsed = round(time.monotonic() - start_time, 1)

        logger.info(
            "Report generation COMPLETE — run_id=%s  severity=%s  elapsed=%.1fs",
            run_id, getattr(report, "severity", "?"), elapsed,
        )

        result = report.to_dict() if hasattr(report, "to_dict") else dict(report)
        result["generation_latency_s"] = elapsed
        return result

    except SoftTimeLimitExceeded:
        logger.warning("Report generation time limit reached.")
        return {
            "status":    "time_limit",
            "run_id":    run_id,
            "elapsed_s": round(time.monotonic() - start_time, 1),
        }
    except Exception as exc:
        logger.error("Report generation FAILED — %s", exc, exc_info=True)
        raise self.retry(exc=exc, countdown=10)


# ══════════════════════════════════════════════════════════════════════════════
# Training summary report
# ══════════════════════════════════════════════════════════════════════════════

@celery_app.task(
    bind          = True,
    base          = _BaseReportTask,
    name          = "tasks.report_tasks.generate_training_summary",
    queue         = "reports",
    max_retries   = 0,
    track_started = True,
    ignore_result = False,
)
def generate_training_summary(
    self:       Task,
    run_id:     str,
    metrics:    Dict[str, Any],
    config:     Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Generate a post-training summary report.

    Creates a structured summary with key metrics, CVaR analysis,
    drift events, and EWC statistics.

    Parameters
    ----------
    run_id : str
        Training run identifier.
    metrics : dict
        Final training metrics from ``agent.get_metrics()``.
    config : dict, optional
        Application configuration.

    Returns
    -------
    dict
        Training summary report dict.
    """
    start_time = time.monotonic()
    cfg = config or _load_config()

    logger.info("Training summary START — run_id=%s", run_id)

    self.update_state(
        state = "STARTED",
        meta  = {"run_id": run_id, "message": "Generating training summary..."},
    )

    try:
        llm_client = self._get_llm_client(cfg)

        from explainability.report_generator import ReportGenerator
        generator = ReportGenerator(llm=llm_client, auto_save=True)

        context = {
            "report_type":       "training_summary",
            "run_id":            run_id,
            "total_timesteps":   metrics.get("total_timesteps", 0),
            "mean_reward":       metrics.get("mean_reward"),
            "cvar_005":          metrics.get("cvar_005"),
            "cvar_001":          metrics.get("cvar_001"),
            "catastrophic_rate": metrics.get("catastrophic_rate"),
            "ewc_tasks":         metrics.get("ewc_tasks_registered", 0),
            "ewc_forgetting":    metrics.get("ewc_forgetting"),
            "drift_events":      metrics.get("drift_events", 0),
            "elapsed_s":         metrics.get("training_elapsed_s", 0),
        }

        report  = generator.generate(context)
        elapsed = round(time.monotonic() - start_time, 1)

        logger.info("Training summary COMPLETE — run_id=%s  elapsed=%.1fs", run_id, elapsed)

        result = report.to_dict() if hasattr(report, "to_dict") else dict(report)
        result["generation_latency_s"] = elapsed
        return result

    except SoftTimeLimitExceeded:
        logger.warning("Training summary generation time limit.")
        return {"status": "time_limit", "run_id": run_id}
    except Exception as exc:
        logger.error("Training summary FAILED — %s", exc, exc_info=True)
        raise


# ── Helper ───────────────────────────────────────────────────────────────────

def _load_config() -> Dict[str, Any]:
    """Load configuration from config.yaml."""
    try:
        from utils.config_loader import load_config
        return load_config("config.yaml")
    except Exception as exc:
        logger.warning("Could not load config.yaml: %s", exc)
        return {}
