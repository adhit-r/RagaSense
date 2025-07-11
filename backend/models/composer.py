from sqlalchemy import Column, Integer, String
from backend.models.base import Base
from sqlalchemy.orm import relationship

class Composer(Base):
    __tablename__ = 'composers'
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False, unique=True) 
    performances = relationship("Performance", back_populates="composer") 