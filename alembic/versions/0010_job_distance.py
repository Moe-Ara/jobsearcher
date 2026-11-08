"""jobs.distance_km: straight-line distance from your home (sorting: priority, then nearest).

Revision ID: 0010
Revises: 0009
Create Date: 2026-11-08

Plain add_column + index on jobs (SQLite ADD COLUMN, no table rebuild — never batch_alter jobs).
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '0010'
down_revision: Union[str, None] = '0009'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('jobs', sa.Column('distance_km', sa.Float(), nullable=True))
    op.create_index('ix_jobs_distance_km', 'jobs', ['distance_km'])


def downgrade() -> None:
    op.drop_index('ix_jobs_distance_km', table_name='jobs')
    op.drop_column('jobs', 'distance_km')
