from typing import Optional, Any
from pydantic import BaseModel

class AudioSampleBase(BaseModel):
    file_path: str
    type: Optional[str] = None
    audio_metadata: Optional[Any] = None
    raga_id: Optional[int] = None
    artist_id: Optional[int] = None

class AudioSampleCreate(AudioSampleBase):
    pass

class AudioSampleRead(AudioSampleBase):
    id: int
    performance_id: Optional[int] = None
    class Config:
        orm_mode = True 