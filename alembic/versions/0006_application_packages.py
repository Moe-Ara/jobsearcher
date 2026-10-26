"""application packages: application agent + reviewer agent runs (tailored CV, letter, reviews, verdict)

Revision ID: 0006
Revises: 0005
Create Date: 2026-10-19

A new table only (no batch_alter on `jobs`, whose FK cascades would delete rows on SQLite table rebuilds).
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '0006'
down_revision: Union[str, None] = '0005'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'application_packages',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('job_id', sa.Integer(), nullable=False),
        sa.Column('status', sa.String(length=20), nullable=False),
        sa.Column('language', sa.String(length=5), nullable=False),
        sa.Column('verdict', sa.String(length=10), nullable=True),
        sa.Column('ats_score', sa.Integer(), nullable=True),
        sa.Column('recruiter_score', sa.Float(), nullable=True),
        sa.Column('rounds', sa.Integer(), nullable=False),
        sa.Column('letter_group_id', sa.String(length=32), nullable=True),
        sa.Column('cv', sa.JSON(), nullable=True),
        sa.Column('email', sa.JSON(), nullable=True),
        sa.Column('reviews', sa.JSON(), nullable=False),
        sa.Column('error', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['job_id'], ['jobs.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_application_packages_job_id', 'application_packages', ['job_id'])
    op.create_index('ix_application_packages_created_at', 'application_packages', ['created_at'])


def downgrade() -> None:
    op.drop_index('ix_application_packages_created_at', table_name='application_packages')
    op.drop_index('ix_application_packages_job_id', table_name='application_packages')
    op.drop_table('application_packages')
