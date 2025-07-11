from sqlalchemy import Column, Integer, String
from app.models.base import Base

class Tala(Base):
    __tablename__ = 'talas'
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False, unique=True) 