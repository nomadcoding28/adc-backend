"""
db/migrations/env.py
=====================
Alembic environment configuration for async SQLAlchemy migrations.

Supports both online (connected to DB) and offline (SQL script generation)
migration modes.

Usage
-----
    # Generate a new migration
    alembic revision --autogenerate -m "add_xyz_table"

    # Apply all pending migrations
    alembic upgrade head
"""

from __future__ import annotations

import asyncio
import logging
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import create_async_engine

from db.base import Base, get_database_url

# Import all models so Alembic can detect them for autogenerate
from db.models.user import User                       # noqa: F401
from db.models.training_run import TrainingRun, Checkpoint  # noqa: F401
from db.models.alert import Alert                     # noqa: F401
from db.models.incident import Incident, IncidentEvent  # noqa: F401
from db.models.evaluation_result import EvaluationResult  # noqa: F401

logger = logging.getLogger(__name__)

# Alembic Config object
config = context.config

# Set up logging from alembic.ini if available
if config.config_file_name:
    try:
        fileConfig(config.config_file_name)
    except Exception:
        pass

# Target metadata for autogenerate
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode.

    Generates SQL scripts without connecting to the database.
    """
    url = get_database_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection) -> None:
    """Run migrations against a live connection (sync callback)."""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    """
    Run migrations in 'online' mode.

    Creates an async engine, then runs migrations inside a connection.
    """
    connectable = create_async_engine(
        get_database_url(),
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
