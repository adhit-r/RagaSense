from sqlalchemy import Column, Integer, String, Text
from sqlalchemy.orm import relationship
from ragasense_backend.app.models.raga import Base

class Type(Base):
    __tablename__ = 'types'

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False, index=True)
    english_name = Column(String(100))
    description = Column(Text)
    region = Column(String, nullable=True)
    # performances = relationship('Performance', back_populates='type')  # Uncomment if you add to Performance 