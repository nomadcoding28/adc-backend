"""
tasks/
======
Celery background task definitions for the ACD Framework.

All long-running operations (RL training, evaluation, KG rebuild,
LLM report generation) run as Celery tasks so the FastAPI server
stays responsive and tasks can be monitored, retried, and revoked.

Task queues
-----------
    training    : RL training runs (exclusive — one at a time, high priority)
    default     : Evaluation, benchmarking, general tasks
    kg          : Knowledge graph rebuild and incremental CVE updates
    reports     : LLM incident report generation

Modules
-------
    celery_app.py         Celery instance, broker config, queue definitions
    training_tasks.py     run_training, run_adversarial_training
    evaluation_tasks.py   run_evaluation, run_benchmark, run_alpha_sweep
    kg_tasks.py           rebuild_kg, update_cves, rebuild_embeddings
    report_tasks.py       generate_report, generate_drift_report,
                          generate_session_summary

Worker commands
---------------
    celery -A tasks.celery_app worker --loglevel=info -Q default,training,kg,reports -c 4
    celery -A tasks.celery_app flower --port=5555
"""

from tasks.celery_app import celery_app

__all__ = ["celery_app"]