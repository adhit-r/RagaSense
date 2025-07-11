from sqlalchemy import Column, Integer, String
from app.models.base import Base

class Composer(Base):
    __tablename__ = 'composers'
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False, unique=True) 