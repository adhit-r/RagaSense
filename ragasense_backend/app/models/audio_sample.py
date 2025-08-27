from sqlalchemy import Column, Integer, String, Text, ForeignKey, LargeBinary
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.declarative import declarative_base
from ragasense_backend.app.models.raga import Base

class AudioSample(Base):
    __tablename__ = 'audio_samples'

    id = Column(Integer, primary_key=True, index=True)
    file_path = Column(String(255), nullable=False)
    type = Column(String(50))  # e.g., 'performance', 'practice', etc.
    audio_metadata = Column(JSONB)
    data = Column(LargeBinary)
    region = Column(String, nullable=True)
    # Foreign keys
    raga_id = Column(Integer, ForeignKey('ragas.id', ondelete='SET NULL'))
    artist_id = Column(Integer, ForeignKey('artists.id', ondelete='SET NULL'))
    song_id = Column(Integer, ForeignKey('songs.id', ondelete='SET NULL'))
    # One-to-one with performance (optional)
    performance = relationship('Performance', back_populates='audio_sample', uselist=False)
    # Relationships
    raga = relationship('Raga', back_populates='audio_samples')
    artist = relationship('Artist', back_populates='audio_samples')
    song = relationship('Song', foreign_keys='AudioSample.song_id', back_populates='audio_samples') 