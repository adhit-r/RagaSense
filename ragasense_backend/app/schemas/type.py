from typing import Optional
from pydantic import BaseModel

class TypeBase(BaseModel):
    name: str
    description: Optional[str] = None

class TypeCreate(TypeBase):
    pass

class TypeRead(TypeBase):
    id: int
    class Config:
        orm_mode = True 