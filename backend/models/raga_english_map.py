from sqlalchemy import Column, Integer, String
from backend.models.raga import Base
 
class RagaEnglishMap(Base):
    __tablename__ = 'raga_english_map'
    id = Column(Integer, primary_key=True)
    english_name = Column(String(100), unique=True, nullable=False) 