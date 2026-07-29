"""SQLAlchemy declarative metadata for persistence models."""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Base class for application persistence models."""
