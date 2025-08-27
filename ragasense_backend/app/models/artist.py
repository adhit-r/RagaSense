from sqlalchemy import Column, Integer, String, Text
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from ragasense_backend.app.models.raga import raga_artist, Base

class Artist(Base):
    __tablename__ = 'artists'

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False, index=True)
    bio = Column(Text)
    country = Column(String(50))
    era = Column(String(50))
    birth_year = Column(Integer)
    death_year = Column(Integer)
    genres = Column(String(200))
    region = Column(String, nullable=True)
    # Relationships
    ragas = relationship('Raga', secondary=raga_artist, back_populates='artists')
    performances = relationship('Performance', back_populates='artist', cascade="all, delete-orphan")
    audio_samples = relationship('AudioSample', back_populates='artist', cascade="all, delete-orphan") 