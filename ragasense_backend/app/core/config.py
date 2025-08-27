import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    PROJECT_NAME: str = "RagaSense"
    DATABASE_URL: str = os.getenv("DATABASE_URL")
    UPLOAD_FOLDER: str = os.getenv("UPLOAD_FOLDER", "uploads/")
    ML_MODEL_PATH: str = os.getenv("ML_MODEL_PATH", "models/raga_classifier/")

settings = Settings() 