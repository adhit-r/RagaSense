from typing import Optional, List
from pydantic import BaseModel

class RagaBase(BaseModel):
    name: str
    arohana: Optional[List[str]] = None
    avarohana: Optional[List[str]] = None
    description: Optional[str] = None
    region: Optional[str] = None
    origin: Optional[str] = None
    language: Optional[str] = None

class RagaCreate(RagaBase):
    pass

class RagaRead(RagaBase):
    id: int
    class Config:
        from_attributes = True 