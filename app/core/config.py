"""Environment-backed application configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass


class ConfigurationError(RuntimeError):
    """Raised when required application configuration is unavailable."""


@dataclass(frozen=True)
class Settings:
    database_url: str


def get_settings() -> Settings:
    """Load the database URL required by database and migration commands."""
    database_url = os.environ.get("DATABASE_URL", "").strip()
    if not database_url:
        raise ConfigurationError("DATABASE_URL must be set.")

    return Settings(database_url=database_url)
