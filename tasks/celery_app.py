"""
tasks/celery_app.py
====================
Celery application instance, broker configuration, queue definitions,
and task routing rules for the ACD Framework.

Architecture
------------
    FastAPI  ──→  run_training.delay()    ──→  [training queue]  ──→  GPU worker
    FastAPI  ──→  run_evaluation.delay()  ──→  [default queue]   ──→  CPU worker
    FastAPI  ──→  rebuild_kg.delay()      ──→  [kg queue]        ──→  CPU worker
    FastAPI  ──→  generate_report.delay() ──→  [reports queue]   ──→  CPU worker

Environment variables
---------------------
    CELERY_BROKER_URL      : redis://localhost:6379/0  (default)
    CELERY_RESULT_BACKEND  : redis://localhost:6379/1  (default)
"""

from __future__ import annotations

import logging
import os

from celery import Celery
from celery.signals import (
    task_failure,
    task_postrun,
    task_prerun,
    task_retry,
    worker_ready,
    worker_shutdown,
)
from kombu import Exchange, Queue

logger = logging.getLogger(__name__)

BROKER_URL  = os.getenv("CELERY_BROKER_URL",     "redis://localhost:6379/0")
BACKEND_URL = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")

celery_app = Celery(
    "acd_framework",
    broker  = BROKER_URL,
    backend = BACKEND_URL,
    include = [
        "tasks.training_tasks",
        "tasks.evaluation_tasks",
        "tasks.kg_tasks",
        "tasks.report_tasks",
    ],
)

_TASK_QUEUES = (
    Queue("training", Exchange("training", type="direct"), routing_key="training",
          queue_arguments={"x-max-priority": 10}),
    Queue("default",  Exchange("default",  type="direct"), routing_key="default"),
    Queue("kg",       Exchange("kg",       type="direct"), routing_key="kg"),
    Queue("reports",  Exchange("reports",  type="direct"), routing_key="reports"),
)

celery_app.conf.update(
    task_serializer              = "json",
    result_serializer            = "json",
    accept_content               = ["json"],
    event_serializer             = "json",
    timezone                     = "UTC",
    enable_utc                   = True,
    task_track_started           = True,
    task_acks_late               = True,
    worker_prefetch_multiplier   = 1,
    task_reject_on_worker_lost   = True,
    task_always_eager            = False,
    result_expires               = 86400,
    result_persistent            = True,
    result_extended              = True,
    task_soft_time_limit         = 7200,
    task_time_limit              = 7500,
    task_default_queue           = "default",
    task_default_exchange        = "default",
    task_default_routing_key     = "default",
    task_queues                  = _TASK_QUEUES,
    task_routes = {
        "tasks.training_tasks.*":   {"queue": "training", "routing_key": "training"},
        "tasks.evaluation_tasks.*": {"queue": "default",  "routing_key": "default"},
        "tasks.kg_tasks.*":         {"queue": "kg",       "routing_key": "kg"},
        "tasks.report_tasks.*":     {"queue": "reports",  "routing_key": "reports"},
    },
    worker_max_tasks_per_child   = 100,
    worker_send_task_events      = True,
    task_annotations = {
        "tasks.kg_tasks.rebuild_kg":            {"rate_limit": "1/h"},
        "tasks.report_tasks.generate_report":   {"rate_limit": "60/m"},
        "tasks.evaluation_tasks.run_benchmark": {"rate_limit": "4/h"},
    },
    beat_schedule = {
        "incremental-cve-update-daily": {
            "task":     "tasks.kg_tasks.update_cves",
            "schedule": 86400,
            "kwargs":   {"days": 1},
            "options":  {"queue": "kg"},
        },
    },
)


@worker_ready.connect
def on_worker_ready(sender, **kwargs) -> None:
    logger.info(
        "Celery worker ready — host=%s, broker=%s",
        getattr(sender, "hostname", "unknown"),
        BROKER_URL.split("@")[-1],
    )


@worker_shutdown.connect
def on_worker_shutdown(sender, **kwargs) -> None:
    logger.info("Celery worker shutting down — host=%s",
                getattr(sender, "hostname", "unknown"))


@task_prerun.connect
def on_task_prerun(task_id, task, args, kwargs, **extra) -> None:
    task_name = getattr(task, "name", "unknown")
    logger.info("Task START — id=%s  name=%s", task_id[:8], task_name)
    try:
        from monitoring.prometheus import PrometheusMetrics
        PrometheusMetrics().record_celery_task(task_name, status="started")
    except Exception:
        pass


@task_postrun.connect
def on_task_postrun(task_id, task, args, kwargs, retval, state, **extra) -> None:
    task_name = getattr(task, "name", "unknown")
    logger.info("Task END — id=%s  name=%s  state=%s", task_id[:8], task_name, state)
    try:
        from monitoring.prometheus import PrometheusMetrics
        PrometheusMetrics().record_celery_task(task_name, status=state.lower())
    except Exception:
        pass


@task_failure.connect
def on_task_failure(task_id, exception, args, kwargs, traceback, einfo, **extra) -> None:
    logger.error(
        "Task FAILED — id=%s  exception=%s: %s",
        (task_id or "?")[:8], type(exception).__name__, str(exception)[:200],
    )
    try:
        from monitoring.sentry import capture_exception, is_initialised
        if is_initialised():
            capture_exception(exception, extra={"task_id": task_id},
                              tags={"component": "celery"})
    except Exception:
        pass


@task_retry.connect
def on_task_retry(request, reason, einfo, **extra) -> None:
    logger.warning(
        "Task RETRY — id=%s  retries=%d  reason=%s",
        str(getattr(request, "id", "?"))[:8],
        getattr(request, "retries", 0),
        reason,
    )