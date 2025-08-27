from sqlalchemy import Column, Integer, String
from app.models.base import Base

class Type(Base):
    __tablename__ = 'types'
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False, unique=True) 