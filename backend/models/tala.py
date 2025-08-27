from sqlalchemy import Column, Integer, String
from backend.models.base import Base
from sqlalchemy.orm import relationship

class Tala(Base):
    __tablename__ = 'talas'
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False, unique=True) 
    performances = relationship("Performance", back_populates="tala") 