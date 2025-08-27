"""Add audio analysis and performance features

Revision ID: 002
Revises: 001
Create Date: 2025-07-05 11:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision = '002'
down_revision = '001'
branch_labels = None
depends_on = None

def upgrade():
    # Add columns for audio analysis features
    op.add_column('ragas', sa.Column('pitch_distribution', JSONB(), nullable=True))
    op.add_column('ragas', sa.Column('tonic_frequency', sa.Float(), nullable=True))
    op.add_column('ragas', sa.Column('characteristic_phrases', JSONB(), nullable=True))
    
    # Add performance and pedagogy related columns
    op.add_column('ragas', sa.Column('aroha_patterns', JSONB(), nullable=True))
    op.add_column('ragas', sa.Column('avaroha_patterns', JSONB(), nullable=True))
    op.add_column('ragas', sa.Column('pakad', sa.Text(), nullable=True))
    op.add_column('ragas', sa.Column('practice_exercises', JSONB(), nullable=True))
    
    # Add metadata columns
    op.add_column('ragas', sa.Column('thaat', sa.String(50), nullable=True))
    op.add_column('ragas', sa.Column('time_period', sa.String(50), nullable=True))
    op.add_column('ragas', sa.Column('regional_style', JSONB(), nullable=True))
    
    # Create index on thaat for faster lookups
    op.create_index('ix_ragas_thaat', 'ragas', ['thaat'])

def downgrade():
    # Remove new columns in reverse order
    op.drop_index('ix_ragas_thaat', 'ragas')
    op.drop_column('ragas', 'regional_style')
    op.drop_column('ragas', 'time_period')
    op.drop_column('ragas', 'thaat')
    op.drop_column('ragas', 'practice_exercises')
    op.drop_column('ragas', 'pakad')
    op.drop_column('ragas', 'avaroha_patterns')
    op.drop_column('ragas', 'aroha_patterns')
    op.drop_column('ragas', 'characteristic_phrases')
    op.drop_column('ragas', 'tonic_frequency')
    op.drop_column('ragas', 'pitch_distribution')
