"""application packages: meta (agent tool log, stage timings, company research)

Revision ID: 0007
Revises: 0006
Create Date: 2026-10-19

Plain add_column on application_packages (SQLite ADD COLUMN, no table rebuild).
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '0007'
down_revision: Union[str, None] = '0006'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('application_packages', sa.Column('meta', sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column('application_packages', 'meta')
