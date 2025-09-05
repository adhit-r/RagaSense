from typing import Optional, List
from pydantic import BaseModel

class PerformanceBase(BaseModel):
    title: Optional[str] = None
    date: Optional[str] = None  # ISO format string
    location: Optional[str] = None
    artist_id: Optional[int] = None
    raga_id: Optional[int] = None
    composer_id: Optional[int] = None
    type_id: Optional[int] = None
    tala_id: Optional[int] = None
    song_id: Optional[int] = None
    audio_sample_id: Optional[int] = None
    duration: Optional[float] = None  # in seconds
    rating: Optional[float] = None
    notes: Optional[str] = None

class PerformanceCreate(PerformanceBase):
    pass

class PerformanceUpdate(PerformanceBase):
    pass

class PerformanceRead(PerformanceBase):
    id: int

    class Config:
        orm_mode = True 