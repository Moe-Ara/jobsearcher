"""application notes: the applicant's own reasons per job + cover-letter strategy

Revision ID: 0005
Revises: 0004
Create Date: 2026-10-12

A new table only (no batch_alter on `jobs`, whose FK cascades would delete rows on SQLite table rebuilds).
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '0005'
down_revision: Union[str, None] = '0004'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'application_notes',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('job_id', sa.Integer(), nullable=False),
        sa.Column('why_interested', sa.Text(), nullable=True),
        sa.Column('why_fit', sa.Text(), nullable=True),
        sa.Column('emphasize', sa.Text(), nullable=True),
        sa.Column('avoid', sa.Text(), nullable=True),
        sa.Column('cover_letter_strategy', sa.String(length=40), nullable=False, server_default='NONE'),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['job_id'], ['jobs.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_application_notes_job_id', 'application_notes', ['job_id'], unique=True)


def downgrade() -> None:
    op.drop_index('ix_application_notes_job_id', table_name='application_notes')
    op.drop_table('application_notes')
