from typing import Optional
from datetime import date
from pydantic import BaseModel

class PerformanceBase(BaseModel):
    date: Optional[date] = None
    venue: Optional[str] = None
    notes: Optional[str] = None
    artist_id: Optional[int] = None
    raga_id: Optional[int] = None
    audio_sample_id: Optional[int] = None
    type_id: Optional[int] = None

class PerformanceCreate(PerformanceBase):
    pass

class PerformanceRead(PerformanceBase):
    id: int
    class Config:
        orm_mode = True 