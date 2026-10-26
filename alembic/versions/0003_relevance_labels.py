"""relevance labels

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-06 17:41:16.925549
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '0003'
down_revision: Union[str, None] = '0002'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Plain ADD COLUMN (no batch/table rebuild): rebuilding `jobs` on SQLite with foreign keys on would cascade-delete
    # dependent rows. ADD COLUMN with a constant default is supported natively by SQLite and PostgreSQL.
    op.add_column('jobs', sa.Column('relevance_label', sa.String(length=10), server_default='UNRATED', nullable=False))
    op.add_column('jobs', sa.Column('relevance_reasons', sa.JSON(none_as_null=True), nullable=True))
    op.add_column('jobs', sa.Column('relevance_labeled_at', sa.DateTime(), nullable=True))
    op.create_index('ix_jobs_relevance_label', 'jobs', ['relevance_label'], unique=False)


def downgrade() -> None:
    op.drop_index('ix_jobs_relevance_label', table_name='jobs')
    for col in ('relevance_labeled_at', 'relevance_reasons', 'relevance_label'):
        op.drop_column('jobs', col)  # SQLite >= 3.35 supports DROP COLUMN natively
