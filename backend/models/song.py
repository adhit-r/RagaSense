from sqlalchemy import Column, Integer, String
from backend.models.base import Base
from sqlalchemy.orm import relationship

class Song(Base):
    __tablename__ = 'songs'
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False) 
    performances = relationship("Performance", back_populates="song") 