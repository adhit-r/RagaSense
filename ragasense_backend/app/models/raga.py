from sqlalchemy import Column, Integer, String, Text, ForeignKey, Table
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

# Association table for many-to-many Raga <-> Artist
raga_artist = Table(
    'raga_artist', Base.metadata,
    Column('raga_id', Integer, ForeignKey('ragas.id', ondelete='CASCADE')),
    Column('artist_id', Integer, ForeignKey('artists.id', ondelete='CASCADE'))
)

class Raga(Base):
    __tablename__ = 'ragas'

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False, index=True)
    alternate_names = Column(JSONB)
    tradition = Column(String(50), index=True)  # 'Hindustani' or 'Carnatic'
    arohana = Column(JSONB)
    avarohana = Column(JSONB)
    characteristic_phrases = Column(JSONB)
    vadi = Column(String(20))
    samvadi = Column(String(20))
    varjya_swaras = Column(JSONB)
    jati = Column(String(50))
    time = Column(JSONB)
    season = Column(JSONB)
    rasa = Column(JSONB)
    mood = Column(JSONB)
    description = Column(Text)
    history = Column(Text)
    notable_compositions = Column(JSONB)
    audio_features = Column(JSONB)
    pitch_distribution = Column(JSONB)
    tonic_frequency = Column(String(20))
    pakad = Column(Text)
    thaat = Column(String(50), index=True)
    melakarta_number = Column(Integer, index=True)
    carnatic_equivalent = Column(String(100))
    hindustani_equivalent = Column(String(100))
    chakra = Column(String(50))
    janya_ragas = Column(JSONB)
    parent_raga_id = Column(Integer, ForeignKey('ragas.id', ondelete='SET NULL'))
    parent_raga = relationship('Raga', remote_side='Raga.id')
    type_id = Column(Integer, ForeignKey('types.id', ondelete='SET NULL'))
    tala_id = Column(Integer, ForeignKey('talas.id', ondelete='SET NULL'))
    region = Column(String, nullable=True)
    origin = Column(String(100))
    language = Column(String(50))
    # Relationships
    artists = relationship('Artist', secondary=raga_artist, back_populates='ragas')
    composers = relationship('Composer', secondary='raga_composer', back_populates='ragas')
    performances = relationship('Performance', back_populates='raga', cascade="all, delete-orphan")
    audio_samples = relationship('AudioSample', back_populates='raga', cascade="all, delete-orphan") 