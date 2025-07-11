from sqlalchemy import Column, Integer, String, JSON
from backend.models.base import Base
from sqlalchemy.orm import relationship

class Artist(Base):
    __tablename__ = "artists"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    stats = Column(JSON, nullable=True)
    poster = Column(JSON, nullable=True)
    info = Column(JSON, nullable=True)
    title = Column(JSON, nullable=True)
    songs = Column(JSON, nullable=True)
    folders = Column(JSON, nullable=True)
    languages = Column(JSON, nullable=True) 
    performances = relationship("Performance", back_populates="artist") 