"""jobs.track: main (your career search) | side (part-time, mini-jobs, temporary work — any field)

Revision ID: 0008
Revises: 0007
Create Date: 2026-10-22

Plain add_column + index on jobs (SQLite ADD COLUMN, no table rebuild — never batch_alter jobs).
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '0008'
down_revision: Union[str, None] = '0007'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('jobs', sa.Column('track', sa.String(length=12), nullable=False, server_default='main'))
    op.create_index('ix_jobs_track', 'jobs', ['track'])


def downgrade() -> None:
    op.drop_index('ix_jobs_track', table_name='jobs')
    op.drop_column('jobs', 'track')
