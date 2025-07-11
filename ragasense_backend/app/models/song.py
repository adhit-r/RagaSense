from sqlalchemy import Column, Integer, String, ForeignKey, Text
from sqlalchemy.orm import relationship
from ragasense_backend.app.models.type import Type
from ragasense_backend.app.models.composer import Composer
from ragasense_backend.app.models.raga import Base, Raga
from ragasense_backend.app.models.tala import Tala
from ragasense_backend.app.models.audio_sample import AudioSample

class Song(Base):
    __tablename__ = 'songs'

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(200), nullable=False, index=True)
    lyrics = Column(Text)
    language = Column(String(50))
    region = Column(String, nullable=True)
    composer_id = Column(Integer, ForeignKey('composers.id', ondelete='SET NULL'))
    raga_id = Column(Integer, ForeignKey('ragas.id', ondelete='SET NULL'))
    type_id = Column(Integer, ForeignKey('types.id', ondelete='SET NULL'))
    tala_id = Column(Integer, ForeignKey('talas.id', ondelete='SET NULL'))
    audio_sample_id = Column(Integer, ForeignKey('audio_samples.id', ondelete='SET NULL'))
    raw_raga_name = Column(String, nullable=True)
    raw_composer_name = Column(String, nullable=True)
    # Relationships
    composer = relationship('Composer')
    raga = relationship('Raga')
    type = relationship('Type')
    tala = relationship('Tala', back_populates='songs')
    audio_sample = relationship('AudioSample', foreign_keys='Song.audio_sample_id')
    audio_samples = relationship('AudioSample', back_populates='song', foreign_keys='AudioSample.song_id')
    performances = relationship('Performance', back_populates='song') 