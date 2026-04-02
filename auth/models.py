"""auth/models.py — User and Role SQLAlchemy ORM models."""

from __future__ import annotations

import time
from typing import Optional

from sqlalchemy import Boolean, Column, Float, Integer, String, Text
from sqlalchemy.orm import relationship

try:
    from db.base import Base
except ImportError:
    from sqlalchemy.orm import DeclarativeBase
    class Base(DeclarativeBase):
        pass


class Role(Base):
    """User role (admin, analyst, viewer)."""
    __tablename__ = "roles"

    id          = Column(Integer, primary_key=True, index=True)
    name        = Column(String(50), unique=True, nullable=False)
    description = Column(String(200), default="")

    def __repr__(self) -> str:
        return f"Role(id={self.id}, name={self.name!r})"


class User(Base):
    """Application user."""
    __tablename__ = "users"

    id           = Column(Integer, primary_key=True, index=True)
    username     = Column(String(100), unique=True, nullable=False, index=True)
    email        = Column(String(200), unique=True, nullable=False, index=True)
    password_hash= Column(String(200), nullable=False)
    role         = Column(String(50), default="analyst")
    is_active    = Column(Boolean, default=True)
    created_at   = Column(Float, default=time.time)
    last_login   = Column(Float, nullable=True)

    def __repr__(self) -> str:
        return f"User(id={self.id}, username={self.username!r}, role={self.role!r})"