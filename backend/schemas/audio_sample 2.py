from typing import Optional
from pydantic import BaseModel

class AudioSampleBase(BaseModel):
    file_path: Optional[str] = None

class AudioSampleCreate(AudioSampleBase):
    pass

class AudioSampleRead(AudioSampleBase):
    id: int
    class Config:
        from_attributes = True 