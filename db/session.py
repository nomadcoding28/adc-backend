"""
db/session.py
=============
Async database session management and FastAPI dependency.

Usage in route handlers
-----------------------
    from fastapi import Depends
    from db.session import get_db
    from sqlalchemy.ext.asyncio import AsyncSession

    @router.get("/users")
    async def list_users(db: AsyncSession = Depends(get_db)):
        result = await db.execute(select(User))
        return result.scalars().all()
"""

from __future__ import annotations

import logging
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession

from db.base import create_engine, create_session_factory, create_tables

logger = logging.getLogger(__name__)

# Module-level engine and session factory (initialised on first use)
_engine          = None
_session_factory = None


def _get_session_factory():
    global _engine, _session_factory
    if _session_factory is None:
        _engine          = create_engine()
        _session_factory = create_session_factory(_engine)
    return _session_factory


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that yields an async database session.

    Automatically commits on success and rolls back on exception.

    Usage
    -----
        async def my_route(db: AsyncSession = Depends(get_db)):
            ...
    """
    factory = _get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db() -> None:
    """Initialise the database (create tables if they don't exist)."""
    engine = create_engine()
    await create_tables(engine)
    logger.info("Database initialised.")


async def get_db_session() -> AsyncSession:
    """
    Return a standalone async session (for use outside FastAPI routes,
    e.g. in Celery tasks).

    Caller is responsible for committing / rolling back / closing.
    """
    factory = _get_session_factory()
    return factory()