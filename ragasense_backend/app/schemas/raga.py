from typing import List, Optional, Any
from pydantic import BaseModel

class RagaBase(BaseModel):
    name: str
    alternate_names: Optional[List[str]] = None
    tradition: Optional[str] = None
    arohana: Optional[List[str]] = None
    avarohana: Optional[List[str]] = None
    characteristic_phrases: Optional[List[str]] = None
    vadi: Optional[str] = None
    samvadi: Optional[str] = None
    varjya_swaras: Optional[List[str]] = None
    jati: Optional[str] = None
    time: Optional[List[str]] = None
    season: Optional[List[str]] = None
    rasa: Optional[List[str]] = None
    mood: Optional[List[str]] = None
    description: Optional[str] = None
    history: Optional[str] = None
    notable_compositions: Optional[List[str]] = None
    audio_features: Optional[Any] = None
    pitch_distribution: Optional[Any] = None
    tonic_frequency: Optional[str] = None
    pakad: Optional[str] = None
    thaat: Optional[str] = None
    melakarta_number: Optional[int] = None
    carnatic_equivalent: Optional[str] = None
    hindustani_equivalent: Optional[str] = None
    chakra: Optional[str] = None
    janya_ragas: Optional[List[str]] = None

class RagaCreate(RagaBase):
    pass

class RagaRead(RagaBase):
    id: int
    artists: Optional[List[int]] = None
    composers: Optional[List[int]] = None
    performances: Optional[List[int]] = None
    audio_samples: Optional[List[int]] = None

    class Config:
        orm_mode = True 