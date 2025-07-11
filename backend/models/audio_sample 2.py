from sqlalchemy import Column, Integer, String
from app.models.base import Base

class AudioSample(Base):
    __tablename__ = 'audio_samples'
    id = Column(Integer, primary_key=True, index=True)
    file_path = Column(String, nullable=True) 