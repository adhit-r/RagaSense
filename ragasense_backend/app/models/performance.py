from sqlalchemy import Column, Integer, String, Text, ForeignKey, Date
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from ragasense_backend.app.models.raga import Base
from ragasense_backend.app.models.type import Type
from ragasense_backend.app.models.composer import Composer
from ragasense_backend.app.models.tala import Tala
from ragasense_backend.app.models.song import Song

class Performance(Base):
    __tablename__ = 'performances'

    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date)
    venue = Column(String(100))
    notes = Column(Text)
    region = Column(String, nullable=True)
    # Foreign keys
    artist_id = Column(Integer, ForeignKey('artists.id', ondelete='SET NULL'))
    raga_id = Column(Integer, ForeignKey('ragas.id', ondelete='SET NULL'))
    audio_sample_id = Column(Integer, ForeignKey('audio_samples.id', ondelete='SET NULL'))
    type_id = Column(Integer, ForeignKey('types.id'))
    composer_id = Column(Integer, ForeignKey('composers.id', ondelete='SET NULL'))
    tala_id = Column(Integer, ForeignKey('talas.id', ondelete='SET NULL'))
    song_id = Column(Integer, ForeignKey('songs.id', ondelete='SET NULL'))
    duration = Column(Integer)  # Duration in seconds
    rating = Column(Integer)    # Optional rating, 1-5 or similar
    # Relationships
    artist = relationship('Artist', back_populates='performances')
    raga = relationship('Raga', back_populates='performances')
    audio_sample = relationship('AudioSample', back_populates='performance')
    type = relationship('Type')
    composer = relationship('Composer')
    tala = relationship('Tala')
    song = relationship('Song', back_populates='performances') 