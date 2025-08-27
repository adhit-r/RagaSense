#!/usr/bin/env python3
"""
Initialize the database with sample raga data.

This script populates the database with sample ragas, artists, and audio samples
for testing and development purposes.
"""

import os
import sys
from datetime import datetime
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import models
from backend.models.raga import Raga
from backend.models.artist import Artist
from backend.models.audio_sample import AudioSample
from backend.models.song import Song
from backend.models.type import Type
from backend.models.tala import Tala
from backend.models.composer import Composer
from backend.core.db import Base, get_db

# Database connection
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://raga_user:raga_pass@localhost:5432/ragasense_db")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """Initialize the database with sample data."""
    # Create all tables
    Base.metadata.create_all(bind=engine)
    
    db = SessionLocal()
    
    try:
        # Clear existing data
        db.execute(text('TRUNCATE TABLE audio_samples CASCADE'))
        db.execute(text('TRUNCATE TABLE songs CASCADE'))
        db.execute(text('TRUNCATE TABLE ragam_english_map CASCADE'))
        db.execute(text('TRUNCATE TABLE ragam CASCADE'))
        db.execute(text('TRUNCATE TABLE artists CASCADE'))
        db.execute(text('TRUNCATE TABLE composers CASCADE'))
        db.execute(text('TRUNCATE TABLE types CASCADE'))
        db.execute(text('TRUNCATE TABLE talas CASCADE'))
        
        # Create sample data
        # 1. Create types
        carnatic = Type(name="Carnatic", description="South Indian classical music")
        hindustani = Type(name="Hindustani", description="North Indian classical music")
        db.add_all([carnatic, hindustani])
        
        # 2. Create talas
        adi_tala = Tala(name="Adi Tala", beats=8, description="8 beat cycle")
        rupaka = Tala(name="Rupaka", beats=6, description="6 beat cycle")
        db.add_all([adi_tala, rupaka])
        
        # 3. Create composers
        thyagaraja = Composer(name="Thyagaraja", period="18th century", type=carnatic)
        muthuswami_dikshitar = Composer(name="Muthuswami Dikshitar", period="18th-19th century", type=carnatic)
        tansen = Composer(name="Mian Tansen", period="16th century", type=hindustani)
        db.add_all([thyagaraja, muthuswami_dikshitar, tansen])
        
        # 4. Create artists
        m_s_subbulakshmi = Artist(
            name="M.S. Subbulakshmi", 
            type=carnatic,
            bio="Legendary Carnatic vocalist"
        )
        bhimsen_joshi = Artist(
            name="Pandit Bhimsen Joshi", 
            type=hindustani,
            bio="Renowned Hindustani classical vocalist"
        )
        db.add_all([m_s_subbulakshmi, bhimsen_joshi])
        
        # 5. Create ragams
        shankarabharanam = Raga(
            name="Shankarabharanam",
            melakarta_number=29,
            arohana=["Sa", "Ri2", "Ga3", "Ma1", "Pa", "Da2", "Ni3", "Sa2"],
            avarohana=["Sa2", "Ni3", "Da2", "Pa", "Ma1", "Ga3", "Ri2", "Sa"],
            type=carnatic,
            time="Evening"
        )
        
        yaman = Raga(
            name="Yaman",
            arohana=["Ni", "Re", "Ga", "Ma#", "Pa", "Dha", "Ni", "Sa2"],
            avarohana=["Sa2", "Ni", "Dha", "Pa", "Ma#", "Ga", "Re", "Sa"],
            type=hindustani,
            time="Evening"
        )
        
        kalyani = Raga(
            name="Kalyani",
            melakarta_number=65,
            arohana=["Sa", "Ri2", "Ga3", "Ma2", "Pa", "Da2", "Ni3", "Sa2"],
            avarohana=["Sa2", "Ni3", "Da2", "Pa", "Ma2", "Ga3", "Ri2", "Sa"],
            type=carnatic,
            time="Evening"
        )
        
        db.add_all([shankarabharanam, yaman, kalyani])
        
        # 6. Create songs
        endaro_mahanubhavulu = Song(
            title="Endaro Mahanubhavulu",
            raga=shankarabharanam,
            composer=thyagaraja,
            tala=adi_tala,
            type=carnatic
        )
        
        bho_shambho = Song(
            title="Bho Shambho",
            raga=yaman,
            composer=tansen,
            type=hindustani
        )
        
        db.add_all([endaro_mahanubhavulu, bho_shambbo])
        
        # 7. Create audio samples
        sample1 = AudioSample(
            file_path="/path/to/endaro_sample1.wav",
            duration=300,
            raga=shankarabharanam,
            artist=m_s_subbulakshmi,
            song=endaro_mahanubhavulu,
            recording_date=datetime(1960, 1, 1),
            type=carnatic
        )
        
        sample2 = AudioSample(
            file_path="/path/to/bho_shambho_sample1.wav",
            duration=420,
            raga=yaman,
            artist=bhimsen_joshi,
            song=bho_shambho,
            recording_date=datetime(1980, 1, 1),
            type=hindustani
        )
        
        db.add_all([sample1, sample2])
        
        # Commit all changes
        db.commit()
        
        print("Successfully initialized database with sample data!")
        
    except Exception as e:
        db.rollback()
        print(f"Error initializing database: {e}")
        raise
    finally:
        db.close()

if __name__ == "__main__":
    print("Initializing database with sample data...")
    init_db()
