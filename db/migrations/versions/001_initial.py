"""
db/migrations/versions/001_initial.py
=======================================
Initial Alembic migration — creates all core tables.

Tables created:
    - users              (authentication)
    - training_runs      (training run history)
    - checkpoints        (saved model checkpoints)
    - alerts             (real-time alert log)
    - incidents          (incident reports)
    - incident_events    (incident timeline events)
    - evaluation_results (evaluation run results)

Revision ID: 001
Revises: None
Create Date: 2026-03-20
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

# Revision identifiers
revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create all initial tables."""

    # ── Users ────────────────────────────────────────────────────────────
    op.create_table(
        "users",
        sa.Column("id",            sa.Integer, primary_key=True, index=True),
        sa.Column("username",      sa.String(100), unique=True, nullable=False, index=True),
        sa.Column("email",         sa.String(200), unique=True, nullable=False),
        sa.Column("password_hash", sa.String(200), nullable=False),
        sa.Column("role",          sa.String(50),  server_default="analyst"),
        sa.Column("is_active",     sa.Boolean,     server_default=sa.text("1")),
        sa.Column("created_at",    sa.Float,       nullable=True),
        sa.Column("last_login",    sa.Float,       nullable=True),
    )

    # ── Training Runs ────────────────────────────────────────────────────
    op.create_table(
        "training_runs",
        sa.Column("id",                sa.Integer,    primary_key=True),
        sa.Column("run_id",            sa.String(50), unique=True, index=True),
        sa.Column("agent_type",        sa.String(50), server_default="cvar_ppo"),
        sa.Column("total_timesteps",   sa.Integer,    server_default="0"),
        sa.Column("final_mean_reward", sa.Float,      nullable=True),
        sa.Column("final_cvar_005",    sa.Float,      nullable=True),
        sa.Column("drift_events",      sa.Integer,    server_default="0"),
        sa.Column("ewc_tasks",         sa.Integer,    server_default="0"),
        sa.Column("elapsed_s",         sa.Float,      server_default="0.0"),
        sa.Column("started_at",        sa.Float,      nullable=True),
        sa.Column("completed_at",      sa.Float,      nullable=True),
        sa.Column("is_running",        sa.Boolean,    server_default=sa.text("0")),
        sa.Column("config_json",       sa.Text,       server_default="{}"),
    )

    # ── Checkpoints ──────────────────────────────────────────────────────
    op.create_table(
        "checkpoints",
        sa.Column("id",          sa.Integer,     primary_key=True),
        sa.Column("run_id",      sa.String(50),  index=True),
        sa.Column("path",        sa.String(500), nullable=False),
        sa.Column("timestep",    sa.Integer,     server_default="0"),
        sa.Column("mean_reward", sa.Float,       nullable=True),
        sa.Column("saved_at",    sa.Float,       nullable=True),
        sa.Column("tag",         sa.String(50),  server_default=""),
    )

    # ── Alerts ───────────────────────────────────────────────────────────
    op.create_table(
        "alerts",
        sa.Column("id",           sa.Integer,    primary_key=True),
        sa.Column("alert_id",     sa.String(20), unique=True, index=True),
        sa.Column("alert_type",   sa.String(30), server_default="INFO"),
        sa.Column("severity",     sa.String(20), server_default="LOW"),
        sa.Column("title",        sa.String(300), nullable=False),
        sa.Column("description",  sa.Text,       server_default=""),
        sa.Column("host",         sa.String(50), nullable=True),
        sa.Column("cve_id",       sa.String(30), nullable=True),
        sa.Column("technique_id", sa.String(20), nullable=True),
        sa.Column("acknowledged", sa.Boolean,    server_default=sa.text("0")),
        sa.Column("timestamp",    sa.Float,      nullable=True),
    )

    # ── Incidents ────────────────────────────────────────────────────────
    op.create_table(
        "incidents",
        sa.Column("id",             sa.Integer,    primary_key=True, index=True),
        sa.Column("incident_id",    sa.String(50), unique=True, index=True, nullable=False),
        sa.Column("title",          sa.String(300), nullable=False),
        sa.Column("severity",       sa.String(20), server_default="MEDIUM"),
        sa.Column("report_type",    sa.String(30), server_default="INCIDENT"),
        sa.Column("cve_ids",        sa.Text,       server_default=""),
        sa.Column("technique_ids",  sa.Text,       server_default=""),
        sa.Column("hosts_affected", sa.Text,       server_default=""),
        sa.Column("markdown",       sa.Text,       server_default=""),
        sa.Column("generated_at",   sa.Float,      nullable=True),
        sa.Column("tokens_used",    sa.Integer,    server_default="0"),
        sa.Column("latency_s",      sa.Float,      server_default="0.0"),
    )

    # ── Incident Events ──────────────────────────────────────────────────
    op.create_table(
        "incident_events",
        sa.Column("id",          sa.Integer,    primary_key=True),
        sa.Column("incident_id", sa.String(50), nullable=False, index=True),
        sa.Column("timestamp",   sa.String(50), nullable=False),
        sa.Column("event",       sa.Text,       nullable=False),
        sa.Column("created_at",  sa.Float,      nullable=True),
    )

    # ── Evaluation Results ───────────────────────────────────────────────
    op.create_table(
        "evaluation_results",
        sa.Column("id",                sa.Integer,    primary_key=True),
        sa.Column("run_id",            sa.String(50), index=True),
        sa.Column("n_episodes",        sa.Integer,    server_default="0"),
        sa.Column("mean_reward",       sa.Float,      nullable=True),
        sa.Column("std_reward",        sa.Float,      nullable=True),
        sa.Column("cvar_005",          sa.Float,      nullable=True),
        sa.Column("success_rate",      sa.Float,      nullable=True),
        sa.Column("catastrophic_rate", sa.Float,      nullable=True),
        sa.Column("ewc_forgetting",    sa.Float,      nullable=True),
        sa.Column("drift_events",      sa.Integer,    server_default="0"),
        sa.Column("elapsed_s",         sa.Float,      server_default="0.0"),
        sa.Column("evaluated_at",      sa.Float,      nullable=True),
        sa.Column("results_json",      sa.Text,       server_default="{}"),
    )


def downgrade() -> None:
    """Drop all tables created in this migration."""
    op.drop_table("evaluation_results")
    op.drop_table("incident_events")
    op.drop_table("incidents")
    op.drop_table("alerts")
    op.drop_table("checkpoints")
    op.drop_table("training_runs")
    op.drop_table("users")
