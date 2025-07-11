from typing import List, Optional
from pydantic import BaseModel

class ComposerBase(BaseModel):
    name: str
    bio: Optional[str] = None
    country: Optional[str] = None
    era: Optional[str] = None

class ComposerCreate(ComposerBase):
    pass

class ComposerRead(ComposerBase):
    id: int
    ragas: Optional[List[int]] = None

    class Config:
        orm_mode = True 