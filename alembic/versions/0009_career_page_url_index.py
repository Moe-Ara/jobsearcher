"""Index jobs.career_page_url — the "same vacancy" lookup (applied_reason) matches on it for every listed job.

Revision ID: 0009
Revises: 0008
Create Date: 2026-10-25

Index only (no table rebuild — never batch_alter jobs).
"""
from typing import Sequence, Union

from alembic import op


revision: str = '0009'
down_revision: Union[str, None] = '0008'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index('ix_jobs_career_page_url', 'jobs', ['career_page_url'])


def downgrade() -> None:
    op.drop_index('ix_jobs_career_page_url', table_name='jobs')
