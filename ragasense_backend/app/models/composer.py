from sqlalchemy import Column, Integer, String, Text, Table, ForeignKey
from sqlalchemy.orm import relationship
from ragasense_backend.app.models.raga import Base

# Association table for many-to-many Raga <-> Composer
dag_composer = Table(
    'raga_composer', Base.metadata,
    Column('raga_id', Integer, ForeignKey('ragas.id', ondelete='CASCADE')),
    Column('composer_id', Integer, ForeignKey('composers.id', ondelete='CASCADE'))
)

class Composer(Base):
    __tablename__ = 'composers'

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False, index=True)
    bio = Column(Text)
    country = Column(String(50))
    era = Column(String(50))
    birth_year = Column(Integer)
    death_year = Column(Integer)
    region = Column(String, nullable=True)
    # Relationships
    ragas = relationship('Raga', secondary='raga_composer', back_populates='composers') 