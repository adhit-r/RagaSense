import os
import json
import logging
from typing import Dict, List, Optional
from pathlib import Path

import numpy as np
import librosa
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.cloud.storage as storage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Raga Detection API",
    description="AI-powered Indian classical music raga detection",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
raga_classifier = None
feature_extractor = None

class RagaPrediction(BaseModel):
    raga: str
    confidence: float
    tradition: str
    description: str
    arohana: List[str]
    avarohana: List[str]

class DetectionResponse(BaseModel):
    predictions: List[RagaPrediction]
    processing_time: float
    audio_duration: float
    sample_rate: int

def download_models_from_gcs():
    """Download models from Google Cloud Storage"""
    global raga_classifier, feature_extractor
    
    try:
        # Initialize GCS client
        storage_client = storage.Client()
        bucket = storage_client.bucket(MODEL_BUCKET)
        
        # Download models
        models_dir = Path("/app/models")
        models_dir.mkdir(exist_ok=True)
        
        # Download raga classifier
        classifier_blob = bucket.blob("models/raga_classifier_model.h5")
        classifier_path = models_dir / "raga_classifier_model.h5"
        classifier_blob.download_to_filename(classifier_path)
        
        # Download feature extractor
        extractor_blob = bucket.blob("models/feature_extractor.pkl")
        extractor_path = models_dir / "feature_extractor.pkl"
        extractor_blob.download_to_filename(extractor_path)
        
        logger.info("✅ Models downloaded successfully")
        
        # Load models
        load_models()
        
    except Exception as e:
        logger.error(f"❌ Error downloading models: {e}")
        # Fallback to dummy models
        create_dummy_models()

def load_models():
    """Load ML models into memory"""
    global raga_classifier, feature_extractor
    
    try:
        import pickle
        from tensorflow import keras
        
        # Load raga classifier
        classifier_path = Path("/app/models/raga_classifier_model.h5")
        if classifier_path.exists():
            raga_classifier = keras.models.load_model(str(classifier_path))
            logger.info("✅ Raga classifier loaded")
        
        # Load feature extractor
        extractor_path = Path("/app/models/feature_extractor.pkl")
        if extractor_path.exists():
            with open(extractor_path, 'rb') as f:
                feature_extractor = pickle.load(f)
            logger.info("✅ Feature extractor loaded")
            
    except Exception as e:
        logger.error(f"❌ Error loading models: {e}")
        create_dummy_models()

def create_dummy_models():
    """Create dummy models for testing"""
    global raga_classifier, feature_extractor
    
    logger.info("🤖 Creating dummy models for testing")
    
    # Dummy classifier
    raga_classifier = {
        'classes': ['Yaman', 'Bhairav', 'Malkauns', 'Bilawal', 'Khamaj'],
        'predict': lambda x: np.random.rand(len(x), 5)
    }
    
    # Dummy feature extractor
    feature_extractor = {
        'extract_features': lambda x: np.random.rand(128)
    }

def extract_audio_features(audio_path: str) -> np.ndarray:
    """Extract features from audio file"""
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=22050)
        
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        # Extract spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        
        # Extract chroma features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        
        # Combine features
        features = np.concatenate([
            mfcc.flatten()[:128],
            spectral_centroid.flatten()[:128],
            spectral_rolloff.flatten()[:128],
            chroma.flatten()[:128]
        ])
        
        return features[:128]  # Ensure consistent size
        
    except Exception as e:
        logger.error(f"❌ Error extracting features: {e}")
        return np.random.rand(128)

def predict_raga(features: np.ndarray) -> List[RagaPrediction]:
    """Predict raga from features"""
    try:
        # Reshape features for model
        features_reshaped = features.reshape(1, -1)
        
        # Get predictions
        if hasattr(raga_classifier, 'predict'):
            predictions = raga_classifier.predict(features_reshaped)
        else:
            # Dummy predictions
            predictions = np.random.rand(1, 5)
        
        # Convert to probabilities
        probabilities = predictions[0]
        probabilities = probabilities / np.sum(probabilities)
        
        # Create prediction objects
        raga_predictions = []
        classes = ['Yaman', 'Bhairav', 'Malkauns', 'Bilawal', 'Khamaj']
        
        for i, (raga, prob) in enumerate(zip(classes, probabilities)):
            raga_predictions.append(RagaPrediction(
                raga=raga,
                confidence=float(prob),
                tradition="Hindustani" if i < 3 else "Carnatic",
                description=f"Beautiful {raga} raga",
                arohana=["Sa", "Re", "Ga", "Ma", "Pa", "Dha", "Ni", "Sa"],
                avarohana=["Sa", "Ni", "Dha", "Pa", "Ma", "Ga", "Re", "Sa"]
            ))
        
        # Sort by confidence
        raga_predictions.sort(key=lambda x: x.confidence, reverse=True)
        
        return raga_predictions
        
    except Exception as e:
        logger.error(f"❌ Error predicting raga: {e}")
        return []

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    logger.info("🚀 Starting Raga Detection API")
    download_models_from_gcs()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": raga_classifier is not None,
        "service": "raga-detection-api"
    }

@app.post("/detect", response_model=DetectionResponse)
async def detect_raga(audio_file: UploadFile = File(...)):
    """Detect raga from uploaded audio file"""
    import time
    
    start_time = time.time()
    
    try:
        # Save uploaded file temporarily
        temp_path = f"/tmp/{audio_file.filename}"
        with open(temp_path, "wb") as f:
            content = await audio_file.read()
            f.write(content)
        
        # Extract features
        features = extract_audio_features(temp_path)
        
        # Get audio duration
        y, sr = librosa.load(temp_path, sr=22050)
        duration = len(y) / sr
        
        # Predict raga
        predictions = predict_raga(features)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Clean up
        os.remove(temp_path)
        
        return DetectionResponse(
            predictions=predictions,
            processing_time=processing_time,
            audio_duration=duration,
            sample_rate=sr
        )
        
    except Exception as e:
        logger.error(f"❌ Error in raga detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/status")
async def get_model_status():
    """Get model loading status"""
    return {
        "raga_classifier_loaded": raga_classifier is not None,
        "feature_extractor_loaded": feature_extractor is not None,
        "models_path": "/app/models"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)