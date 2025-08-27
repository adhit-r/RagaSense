from sqlalchemy import Column, Integer, String
from app.models.base import Base

class Song(Base):
    __tablename__ = 'songs'
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False) 