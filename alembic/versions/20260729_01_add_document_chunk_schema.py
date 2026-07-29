"""Add document and chunk persistence schema.

Revision ID: 20260729_01
Revises: 20260728_01
Create Date: 2026-07-29
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

revision = "20260729_01"
down_revision = "20260728_01"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "documents",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("filename", sa.String(length=255), nullable=False),
        sa.Column("content_checksum", sa.String(length=64), nullable=False),
        sa.Column("media_type", sa.String(length=255), nullable=False),
        sa.Column(
            "ingested_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "content_checksum ~ '^[0-9a-f]{64}$'",
            name="ck_documents_content_checksum_sha256",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("content_checksum", name="uq_documents_content_checksum"),
    )
    op.create_table(
        "chunks",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("document_id", sa.UUID(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("page_number", sa.Integer(), nullable=True),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(1536), nullable=False),
        sa.CheckConstraint(
            "chunk_index >= 0", name="ck_chunks_chunk_index_non_negative"
        ),
        sa.CheckConstraint(
            "page_number IS NULL OR page_number >= 0",
            name="ck_chunks_page_number_non_negative",
        ),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "document_id",
            "chunk_index",
            name="uq_chunks_document_id_chunk_index",
        ),
    )
    op.create_index(
        "ix_chunks_embedding_hnsw_cosine",
        "chunks",
        ["embedding"],
        unique=False,
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )


def downgrade() -> None:
    op.drop_index("ix_chunks_embedding_hnsw_cosine", table_name="chunks")
    op.drop_table("chunks")
    op.drop_table("documents")
