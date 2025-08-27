from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime, Text
from sqlalchemy.orm import relationship
from backend.models.base import Base
from backend.models.song import Song
from backend.models.artist import Artist
from backend.models.raga import Raga
from backend.models.composer import Composer
from backend.models.type import Type
from backend.models.tala import Tala
from backend.models.audio_sample import AudioSample

class Performance(Base):
    __tablename__ = 'performances'

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=True)
    date = Column(String, nullable=True)  # ISO format string
    location = Column(String, nullable=True)
    artist_id = Column(Integer, ForeignKey('artists.id'), nullable=True)
    raga_id = Column(Integer, ForeignKey('ragas.id'), nullable=True)
    composer_id = Column(Integer, ForeignKey('composers.id'), nullable=True)
    type_id = Column(Integer, ForeignKey('types.id'), nullable=True)
    tala_id = Column(Integer, ForeignKey('talas.id'), nullable=True)
    song_id = Column(Integer, ForeignKey('songs.id'), nullable=True)
    audio_sample_id = Column(Integer, ForeignKey('audio_samples.id'), nullable=True)
    duration = Column(Float, nullable=True)
    rating = Column(Float, nullable=True)
    notes = Column(Text, nullable=True)

    artist = relationship('Artist', back_populates='performances')
    raga = relationship('Raga', back_populates='performances')
    composer = relationship('Composer', back_populates='performances')
    type = relationship('Type', back_populates='performances')
    tala = relationship('Tala', back_populates='performances')
    song = relationship('Song', back_populates='performances')
    audio_sample = relationship('AudioSample', back_populates='performances') 