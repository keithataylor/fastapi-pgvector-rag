from __future__ import annotations

import os

import pytest
from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine, text

from app.core.config import get_settings


pytestmark = pytest.mark.migration_integration


def _require_migration_integration() -> None:
    if os.environ.get("RUN_MIGRATION_INTEGRATION") != "1":
        pytest.skip(
            "set RUN_MIGRATION_INTEGRATION=1 to run migration integration tests"
        )


def _vector_extension_is_enabled(database_url: str) -> bool:
    engine = create_engine(database_url)
    try:
        with engine.connect() as connection:
            return bool(
                connection.execute(
                    text(
                        "SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector')"
                    )
                ).scalar_one()
            )
    finally:
        engine.dispose()


def _current_revision(database_url: str) -> str | None:
    engine = create_engine(database_url)
    try:
        with engine.connect() as connection:
            return MigrationContext.configure(connection).get_current_revision()
    finally:
        engine.dispose()


def test_pgvector_extension_migration_lifecycle() -> None:
    _require_migration_integration()
    database_url = get_settings().database_url
    alembic_config = Config("alembic.ini")
    script = ScriptDirectory.from_config(alembic_config)

    command.upgrade(alembic_config, "head")
    assert _vector_extension_is_enabled(database_url)

    command.downgrade(alembic_config, "base")
    assert not _vector_extension_is_enabled(database_url)

    command.upgrade(alembic_config, "head")
    assert _vector_extension_is_enabled(database_url)
    assert _current_revision(database_url) == script.get_current_head()
