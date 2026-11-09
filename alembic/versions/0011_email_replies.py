"""email_messages: replies from employers (rejection / interview / next step / received), the job they belong to.

Revision ID: 0011
Revises: 0010
Create Date: 2026-11-09

Plain add_column + index on email_messages (SQLite ADD COLUMN, no table rebuild).
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '0011'
down_revision: Union[str, None] = '0010'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('email_messages', sa.Column('reply_kind', sa.String(length=20), nullable=True))
    # Inline REFERENCES: valid ADD COLUMN on SQLite (no constraint ALTER) and PostgreSQL alike.
    op.execute("ALTER TABLE email_messages ADD COLUMN job_id INTEGER REFERENCES jobs(id) ON DELETE SET NULL")
    op.add_column('email_messages', sa.Column('snippet', sa.Text(), nullable=True))
    op.add_column('email_messages', sa.Column('effect', sa.String(length=80), nullable=True))
    op.add_column('email_messages', sa.Column('match_note', sa.String(length=300), nullable=True))
    op.create_index('ix_email_messages_reply_kind', 'email_messages', ['reply_kind'])
    op.create_index('ix_email_messages_job_id', 'email_messages', ['job_id'])


def downgrade() -> None:
    op.drop_index('ix_email_messages_job_id', table_name='email_messages')
    op.drop_index('ix_email_messages_reply_kind', table_name='email_messages')
    for col in ('match_note', 'effect', 'snippet', 'job_id', 'reply_kind'):
        op.drop_column('email_messages', col)
