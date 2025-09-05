from typing import Optional, List, Dict, Any
from pydantic import BaseModel

class Stat(BaseModel):
    H: str
    C: str

class Poster(BaseModel):
    P: str

class Info(BaseModel):
    H: str
    V: str

class Title(BaseModel):
    T: int
    H: str
    V: str
    I: str

class SongRef(BaseModel):
    I: str
    T: int
    S: int
    R: int
    C: int
    A: int
    D: str
    V: str
    J: str

class ArtistBase(BaseModel):
    stats: Optional[List[Stat]] = None
    poster: Optional[Poster] = None
    info: Optional[List[Info]] = None
    title: Optional[Title] = None
    songs: Optional[List[SongRef]] = None
    folders: Optional[List[List[Any]]] = None
    languages: Optional[Dict[str, str]] = None

class ArtistCreate(ArtistBase):
    pass

class ArtistRead(ArtistBase):
    id: int
    class Config:
        from_attributes = True 