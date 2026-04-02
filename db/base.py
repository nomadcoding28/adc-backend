"""
db/base.py
==========
SQLAlchemy declarative base and engine/session factory.

Supports both SQLite (dev/testing) and PostgreSQL (production).

Configuration (via environment variables or config.yaml)
---------------------------------------------------------
    DATABASE_URL    : Full SQLAlchemy URL.
                      SQLite:     sqlite+aiosqlite:///./acd.db
                      PostgreSQL: postgresql+asyncpg://user:pass@host/db
"""

from __future__ import annotations

import logging
import os
from typing import Any

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

logger = logging.getLogger(__name__)

# ── Declarative base (all models inherit from this) ─────────────────────────
class Base(DeclarativeBase):
    """SQLAlchemy declarative base for all ACD ORM models."""

    def to_dict(self) -> dict:
        """Convert ORM instance to a plain dict."""
        return {
            col.name: getattr(self, col.name)
            for col in self.__table__.columns
        }


# ── Engine factory ───────────────────────────────────────────────────────────
_DEFAULT_URL = "sqlite+aiosqlite:///./acd.db"

def get_database_url() -> str:
    """Return the database URL from env or default."""
    return os.getenv("DATABASE_URL", _DEFAULT_URL)


def create_engine(database_url: str = None) -> AsyncEngine:
    """
    Create and return an async SQLAlchemy engine.

    Parameters
    ----------
    database_url : str, optional
        SQLAlchemy URL.  Defaults to DATABASE_URL env var or SQLite.

    Returns
    -------
    AsyncEngine
    """
    url = database_url or get_database_url()

    kwargs: dict[str, Any] = {"echo": False}

    # SQLite doesn't support connection pool settings
    if "sqlite" not in url:
        kwargs.update({
            "pool_size":     10,
            "max_overflow":  20,
            "pool_pre_ping": True,
            "pool_recycle":  3600,
        })

    engine = create_async_engine(url, **kwargs)
    logger.info("Database engine created — %s", url.split("@")[-1])  # hide credentials
    return engine


def create_session_factory(engine: AsyncEngine) -> sessionmaker:
    """Return an async session factory bound to the given engine."""
    return sessionmaker(
        bind        = engine,
        class_      = AsyncSession,
        expire_on_commit = False,
        autocommit  = False,
        autoflush   = False,
    )


async def create_tables(engine: AsyncEngine) -> None:
    """Create all tables (idempotent — safe to call on startup)."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables created / verified.")