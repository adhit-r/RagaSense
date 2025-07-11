"""Add Carnatic-Hindustani distinction

Revision ID: 003
Revises: 002
Create Date: 2025-07-05 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision = '003'
down_revision = '002'
branch_labels = None
depends_on = None

def upgrade():
    # Add columns for tradition-specific features
    op.add_column('ragas', sa.Column('tradition', sa.String(50), nullable=True))
    op.add_column('ragas', sa.Column('melakarta_number', sa.Integer(), nullable=True))
    op.add_column('ragas', sa.Column('carnatic_equivalent', sa.String(100), nullable=True))
    op.add_column('ragas', sa.Column('hindustani_equivalent', sa.String(100), nullable=True))
    op.add_column('ragas', sa.Column('janaka_raga', sa.String(100), nullable=True))
    op.add_column('ragas', sa.Column('janya_ragas', JSONB(), nullable=True))
    op.add_column('ragas', sa.Column('chakra', sa.String(50), nullable=True))
    
    # Create indexes
    op.create_index('ix_ragas_tradition', 'ragas', ['tradition'])
    op.create_index('ix_ragas_melakarta_number', 'ragas', ['melakarta_number'])

def downgrade():
    # Remove indexes
    op.drop_index('ix_ragas_melakarta_number', 'ragas')
    op.drop_index('ix_ragas_tradition', 'ragas')
    
    # Remove columns
    op.drop_column('ragas', 'chakra')
    op.drop_column('ragas', 'janya_ragas')
    op.drop_column('ragas', 'janaka_raga')
    op.drop_column('ragas', 'hindustani_equivalent')
    op.drop_column('ragas', 'carnatic_equivalent')
    op.drop_column('ragas', 'melakarta_number')
    op.drop_column('ragas', 'tradition')
