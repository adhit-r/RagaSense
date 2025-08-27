"""Initial schema

Revision ID: 001
Revises: 
Create Date: 2025-07-05 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    # Create ragas table
    op.create_table(
        'ragas',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(length=100), nullable=False),
        sa.Column('alternate_names', JSONB(), nullable=True),
        sa.Column('arohana', JSONB(), nullable=True),
        sa.Column('avarohana', JSONB(), nullable=True),
        sa.Column('vadi', sa.String(length=20), nullable=True),
        sa.Column('samvadi', sa.String(length=20), nullable=True),
        sa.Column('varjya_swaras', JSONB(), nullable=True),
        sa.Column('jati', sa.String(length=50), nullable=True),
        sa.Column('time', JSONB(), nullable=True),
        sa.Column('season', JSONB(), nullable=True),
        sa.Column('rasa', JSONB(), nullable=True),
        sa.Column('mood', JSONB(), nullable=True),
        sa.Column('parent_raga_id', sa.Integer(), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('history', sa.Text(), nullable=True),
        sa.Column('notable_compositions', JSONB(), nullable=True),
        sa.Column('audio_features', JSONB(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.ForeignKeyConstraint(['parent_raga_id'], ['ragas.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('name')
    )
    
    # Create index on name for faster lookups
    op.create_index('ix_ragas_name', 'ragas', ['name'])
    
    # Create raga relationships table
    op.create_table(
        'raga_relationships',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('raga_id', sa.Integer(), nullable=True),
        sa.Column('related_raga_id', sa.Integer(), nullable=True),
        sa.Column('relationship_type', sa.String(length=50), nullable=True),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(['raga_id'], ['ragas.id'], ),
        sa.ForeignKeyConstraint(['related_raga_id'], ['ragas.id'], ),
        sa.PrimaryKeyConstraint('id')
    )

def downgrade():
    op.drop_table('raga_relationships')
    op.drop_index('ix_ragas_name', 'ragas')
    op.drop_table('ragas')
