from sqlalchemy import Column, Integer, String, Float, Boolean, ForeignKey, DateTime, Text
from sqlalchemy.dialects.postgresql import JSONB
from backend.models.base import Base
from sqlalchemy.orm import relationship

class Raga(Base):
    __tablename__ = 'ragas'

    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False, index=True)
    alternate_names = Column(JSONB)
    tradition = Column(String(50), index=True)  # 'Hindustani' or 'Carnatic'
    
    # Scale information
    arohana = Column(JSONB)  # Ascending scale
    avarohana = Column(JSONB)  # Descending scale
    characteristic_phrases = Column(JSONB)  # Characteristic patterns
    
    # Characteristics
    vadi = Column(String(20))  # King note
    samvadi = Column(String(20))  # Queen note
    varjya_swaras = Column(JSONB)  # Omitted notes
    jati = Column(String(50))  # Classification (e.g., Sampoorna, Shadava)
    
    # Performance context
    time = Column(JSONB)  # Time of day (e.g., "Morning", "Evening")
    season = Column(JSONB)  # Season (e.g., "Monsoon", "Summer")
    
    # Emotional content
    rasa = Column(JSONB)  # Emotional essence (e.g., "Shringara", "Bhakti")
    mood = Column(JSONB)  # Mood/feeling
    
    # Additional metadata
    description = Column(Text)
    history = Column(Text)
    notable_compositions = Column(JSONB)
    
    # Audio features (for ML model)
    audio_features = Column(JSONB)
    pitch_distribution = Column(JSONB)
    tonic_frequency = Column(Float)
    aroha_patterns = Column(JSONB)
    avaroha_patterns = Column(JSONB)
    pakad = Column(Text)  # Characteristic catch phrase
    practice_exercises = Column(JSONB)
    
    # Tradition-specific fields
    thaat = Column(String(50), index=True)  # Hindustani classification
    time_period = Column(String(50))
    regional_style = Column(JSONB)
    melakarta_number = Column(Integer, index=True)  # Carnatic classification
    carnatic_equivalent = Column(String(100))
    hindustani_equivalent = Column(String(100))
    janaka_raga = Column(String(100))
    janya_ragas = Column(JSONB)
    chakra = Column(String(50))
    
    # Legacy fields for compatibility
    icon = Column(String(100))
    melakarta_name = Column(String(100))
    stats = Column(JSONB)
    info = Column(JSONB)
    songs = Column(JSONB)
    keyboard = Column(JSONB) 
    performances = relationship("Performance", back_populates="raga") 