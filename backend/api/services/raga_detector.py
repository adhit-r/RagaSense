import os
import librosa
import numpy as np
import tensorflow as tf
from typing import Dict, List, Optional, Tuple
import joblib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class RagaClassifier:
    """
    Raga classification service that handles model loading, feature extraction,
    and prediction.
    """
    
    def __init__(self, model_dir: str = 'ml/models'):
        """
        Initialize the Raga Classifier with model paths.
        
        Args:
            model_dir: Directory containing the model files
        """
        self.model_dir = Path(model_dir)
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.raga_info = {}
        self._load_models()
    
    def _load_models(self) -> None:
        """Load the trained model, scaler, and label encoder."""
        try:
            # Load the latest model version
            model_path = self.model_dir / 'raga_model.h5'
            scaler_path = self.model_dir / 'scaler.pkl'
            encoder_path = self.model_dir / 'label_encoder.pkl'
            
            if not all([model_path.exists(), scaler_path.exists(), encoder_path.exists()]):
                raise FileNotFoundError("Required model files not found. Please train the model first.")
            
            self.model = tf.keras.models.load_model(model_path, compile=False)
            self.scaler = joblib.load(scaler_path)
            self.label_encoder = joblib.load(encoder_path)
            
            # Initialize raga info (this could be loaded from a config file)
            self._init_raga_info()
            
            logger.info("Successfully loaded RagaClassifier model and dependencies")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def _init_raga_info(self) -> None:
        """Initialize raga information dictionary."""
        self.raga_info = {
            'Yaman': {
                'aroha': ['Sa', 'Re', 'Ga', 'Ma#', 'Pa', 'Dha', 'Ni'],
                'avaroha': ['Sa', 'Ni', 'Dha', 'Pa', 'Ma#', 'Ga', 'Re', 'Sa'],
                'vadi': 'Ga',
                'samvadi': 'Ni',
                'time': 'Evening',
                'mood': 'Romantic, Devotional'
            },
            # Add more ragas as needed
        }
    
    def extract_features(self, audio_path: str, duration: int = 30) -> np.ndarray:
        """
        Extract comprehensive audio features for raga classification.
        
        Args:
            audio_path: Path to the audio file
            duration: Maximum duration in seconds to process
            
        Returns:
            numpy.ndarray: Extracted features
        """
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=None, duration=duration)
            
            if len(y) == 0:
                raise ValueError("Empty audio file")
                
            # Ensure audio is mono
            if len(y.shape) > 1:
                y = librosa.to_mono(y)
            
            # Extract features
            features = []
            
            # 1. MFCCs
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            features.append(np.mean(mfccs.T, axis=0))
            
            # 2. Chroma features
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            features.append(np.mean(chroma.T, axis=0))
            
            # 3. Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            
            features.extend([
                np.mean(spectral_centroids),
                np.std(spectral_centroids),
                np.mean(spectral_rolloff),
                np.std(spectral_rolloff),
                np.mean(spectral_bandwidth),
                np.std(spectral_bandwidth)
            ])
            
            # 4. Zero-crossing rate
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            features.extend([np.mean(zcr), np.std(zcr)])
            
            # 5. RMS energy
            rms = librosa.feature.rms(y=y)[0]
            features.extend([np.mean(rms), np.std(rms)])
            
            # 6. Spectral contrast
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            features.extend(np.mean(contrast, axis=1))
            
            # Convert to numpy array and reshape for the model
            features = np.concatenate(features).flatten()
            features = features.reshape(1, -1)
            
            # Scale features
            if self.scaler is not None:
                features = self.scaler.transform(features)
                
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {str(e)}")
            raise
    
    def predict(self, audio_path: str) -> Dict:
        """
        Predict the raga for the given audio file.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            dict: Prediction results with probabilities
        """
        try:
            if self.model is None or self.label_encoder is None:
                raise RuntimeError("Model not loaded. Please initialize the classifier first.")
            
            # Extract features
            features = self.extract_features(audio_path)
            
            # Make prediction
            predictions = self.model.predict(features, verbose=0)
            
            # Get top predictions
            top_k = 3
            top_indices = np.argsort(predictions[0])[-top_k:][::-1]
            
            # Create results
            results = {
                'predictions': [],
                'metadata': {}
            }
            
            for idx in top_indices:
                raga_name = self.label_encoder.inverse_transform([idx])[0]
                proba = float(predictions[0][idx])
                
                results['predictions'].append({
                    'raga': raga_name,
                    'probability': proba,
                    'info': self.raga_info.get(raga_name, {})
                })
            
            # Add metadata
            results['metadata'] = {
                'model_version': '1.0.0',
                'timestamp': str(datetime.utcnow())
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise

# Singleton instance
classifier = RagaClassifier()
