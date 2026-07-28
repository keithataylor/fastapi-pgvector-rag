"""Enable the pgvector extension.

Revision ID: 20260728_01
Revises:
Create Date: 2026-07-28
"""

from __future__ import annotations

from alembic import op

revision = "20260728_01"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")


def downgrade() -> None:
    op.execute("DROP EXTENSION IF EXISTS vector")
