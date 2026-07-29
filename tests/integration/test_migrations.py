from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from collections.abc import Callable, Iterator

import pytest
from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

from app.core.config import get_settings


pytestmark = pytest.mark.migration_integration


def _require_migration_integration() -> None:
    if os.environ.get("RUN_MIGRATION_INTEGRATION") != "1":
        pytest.skip(
            "set RUN_MIGRATION_INTEGRATION=1 to run migration integration tests"
        )


def _migration_test_database_url() -> str:
    raw_url = os.environ.get("MIGRATION_TEST_DATABASE_URL", "").strip()
    if not raw_url:
        pytest.fail("MIGRATION_TEST_DATABASE_URL must be set for migration tests")

    migration_url = make_url(raw_url)
    if migration_url.database in {None, "", "postgres"}:
        pytest.fail("MIGRATION_TEST_DATABASE_URL must target a disposable database")

    _ensure_migration_test_database(raw_url)
    _assert_distinct_database_identities(get_settings().database_url, raw_url)
    return raw_url


def _actual_database_identity(database_url: str) -> tuple[str, str]:
    engine = create_engine(database_url)
    try:
        with engine.connect() as connection:
            result = connection.execute(
                text(
                    "SELECT current_database(), system_identifier::text "
                    "FROM pg_control_system()"
                )
            )
            return tuple(result.one())
    finally:
        engine.dispose()


def _assert_distinct_database_identities(
    application_database_url: str,
    migration_test_database_url: str,
    identity_lookup: Callable[[str], tuple[str, str]] | None = None,
) -> None:
    lookup = identity_lookup or _actual_database_identity
    if lookup(application_database_url) == lookup(migration_test_database_url):
        pytest.fail("MIGRATION_TEST_DATABASE_URL must target a different database")


def _quoted_identifier(identifier: str) -> str:
    return f'"{identifier.replace(chr(34), chr(34) * 2)}"'


def _ensure_migration_test_database(database_url: str) -> None:
    migration_url = make_url(database_url)
    database_name = migration_url.database
    assert database_name is not None
    admin_url = migration_url.set(database="postgres")
    engine = create_engine(admin_url, isolation_level="AUTOCOMMIT")
    try:
        with engine.connect() as connection:
            exists = connection.execute(
                text("SELECT EXISTS (SELECT 1 FROM pg_database WHERE datname = :name)"),
                {"name": database_name},
            ).scalar_one()
            if not exists:
                connection.execute(
                    text(f"CREATE DATABASE {_quoted_identifier(database_name)}")
                )
    finally:
        engine.dispose()


def _recreate_migration_test_database(database_url: str) -> None:
    migration_url = make_url(database_url)
    database_name = migration_url.database
    assert database_name is not None
    admin_url = migration_url.set(database="postgres")
    engine = create_engine(admin_url, isolation_level="AUTOCOMMIT")
    try:
        with engine.connect() as connection:
            identifier = _quoted_identifier(database_name)
            connection.execute(
                text(f"DROP DATABASE IF EXISTS {identifier} WITH (FORCE)")
            )
            connection.execute(text(f"CREATE DATABASE {identifier}"))
    finally:
        engine.dispose()


def _prepare_migration_test_database() -> str:
    database_url = _migration_test_database_url()
    _recreate_migration_test_database(database_url)
    return database_url


@contextmanager
def _use_database_url(database_url: str) -> Iterator[None]:
    original_url = os.environ.get("DATABASE_URL")
    os.environ["DATABASE_URL"] = database_url
    try:
        yield
    finally:
        if original_url is None:
            del os.environ["DATABASE_URL"]
        else:
            os.environ["DATABASE_URL"] = original_url


def _scalar(database_url: str, statement: str, **parameters: object) -> object:
    engine = create_engine(database_url)
    try:
        with engine.connect() as connection:
            return connection.execute(text(statement), parameters).scalar_one()
    finally:
        engine.dispose()


def _vector_extension_is_enabled(database_url: str) -> bool:
    return bool(
        _scalar(
            database_url,
            "SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector')",
        )
    )


def _schema_exists(database_url: str) -> bool:
    return bool(
        _scalar(
            database_url,
            "SELECT to_regclass('public.documents') IS NOT NULL "
            "AND to_regclass('public.chunks') IS NOT NULL",
        )
    )


def _embedding_type(database_url: str) -> str:
    return str(
        _scalar(
            database_url,
            "SELECT format_type(attribute.atttypid, attribute.atttypmod) "
            "FROM pg_attribute AS attribute "
            "JOIN pg_class AS relation ON relation.oid = attribute.attrelid "
            "WHERE relation.relname = 'chunks' "
            "AND attribute.attname = 'embedding' "
            "AND NOT attribute.attisdropped",
        )
    )


def _index_definition(database_url: str) -> tuple[str, str]:
    engine = create_engine(database_url)
    try:
        with engine.connect() as connection:
            result = connection.execute(
                text(
                    "SELECT access_method.amname, operator_class.opcname "
                    "FROM pg_index AS index "
                    "JOIN pg_class AS index_relation ON index_relation.oid = index.indexrelid "
                    "JOIN pg_am AS access_method ON access_method.oid = index_relation.relam "
                    "JOIN pg_opclass AS operator_class ON operator_class.oid = index.indclass[0] "
                    "WHERE index_relation.relname = 'ix_chunks_embedding_hnsw_cosine'"
                )
            )
            return tuple(result.one())
    finally:
        engine.dispose()


def _constraint_names(database_url: str) -> set[str]:
    engine = create_engine(database_url)
    try:
        with engine.connect() as connection:
            return set(
                connection.execute(
                    text(
                        "SELECT constraint_name "
                        "FROM information_schema.table_constraints "
                        "WHERE table_name IN ('documents', 'chunks')"
                    )
                ).scalars()
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


def _assert_schema(database_url: str) -> None:
    assert _schema_exists(database_url)
    assert _embedding_type(database_url) == "vector(1536)"
    assert _index_definition(database_url) == ("hnsw", "vector_cosine_ops")
    assert {
        "uq_documents_content_checksum",
        "uq_chunks_document_id_chunk_index",
        "ck_documents_content_checksum_sha256",
        "ck_chunks_chunk_index_non_negative",
        "ck_chunks_page_number_non_negative",
    } <= _constraint_names(database_url)


def test_database_identity_guard_rejects_semantically_equivalent_urls() -> None:
    application_database_url = "postgresql+psycopg://rag:password@app-host:5432/rag"
    migration_test_database_url = (
        "postgresql+psycopg://rag:password@migration-alias:5432/rag"
    )
    looked_up_urls: list[str] = []

    def identity_lookup(database_url: str) -> tuple[str, str]:
        looked_up_urls.append(database_url)
        return ("rag", "shared-system-identifier")

    with pytest.raises(pytest.fail.Exception, match="must target a different database"):
        _assert_distinct_database_identities(
            application_database_url,
            migration_test_database_url,
            identity_lookup,
        )
    assert looked_up_urls == [application_database_url, migration_test_database_url]


def test_identity_lookup_failure_blocks_database_recreation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "DATABASE_URL", "postgresql+psycopg://rag:password@app-host:5432/rag"
    )
    monkeypatch.setenv(
        "MIGRATION_TEST_DATABASE_URL",
        "postgresql+psycopg://rag:password@migration-host:5432/rag_migration_test",
    )
    module = sys.modules[__name__]
    monkeypatch.setattr(module, "_ensure_migration_test_database", lambda _: None)

    def identity_lookup_failure(_: str) -> tuple[str, str]:
        raise RuntimeError("database identity lookup failed")

    monkeypatch.setattr(module, "_actual_database_identity", identity_lookup_failure)
    recreated_databases: list[str] = []
    monkeypatch.setattr(
        module,
        "_recreate_migration_test_database",
        lambda database_url: recreated_databases.append(database_url),
    )

    with pytest.raises(RuntimeError, match="identity lookup failed"):
        _prepare_migration_test_database()
    assert not recreated_databases


def test_pgvector_extension_and_schema_migration_lifecycle() -> None:
    _require_migration_integration()
    database_url = _prepare_migration_test_database()
    alembic_config = Config("alembic.ini")
    script = ScriptDirectory.from_config(alembic_config)

    with _use_database_url(database_url):
        command.upgrade(alembic_config, "head")
        assert _vector_extension_is_enabled(database_url)
        _assert_schema(database_url)

        command.downgrade(alembic_config, "20260728_01")
        assert _vector_extension_is_enabled(database_url)
        assert not _schema_exists(database_url)

        command.upgrade(alembic_config, "head")
        _assert_schema(database_url)

        command.downgrade(alembic_config, "base")
        assert not _vector_extension_is_enabled(database_url)
        assert not _schema_exists(database_url)

        command.upgrade(alembic_config, "head")
        assert _vector_extension_is_enabled(database_url)
        _assert_schema(database_url)
        assert _current_revision(database_url) == script.get_current_head()
