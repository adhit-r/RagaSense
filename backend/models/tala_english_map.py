from sqlalchemy import Column, Integer, String
from backend.models.raga import Base

class TalaEnglishMap(Base):
    __tablename__ = 'tala_english_map'
    id = Column(Integer, primary_key=True)
    english_name = Column(String(100), unique=True, nullable=False) 