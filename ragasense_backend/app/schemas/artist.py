from typing import List, Optional
from pydantic import BaseModel

class ArtistBase(BaseModel):
    name: str
    bio: Optional[str] = None
    country: Optional[str] = None
    era: Optional[str] = None

class ArtistCreate(ArtistBase):
    pass

class ArtistRead(ArtistBase):
    id: int
    ragas: Optional[List[int]] = None
    performances: Optional[List[int]] = None
    audio_samples: Optional[List[int]] = None

    class Config:
        orm_mode = True 