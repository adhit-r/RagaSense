import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

# Add the ml directory to the path
ml_path = Path(__file__).parent.parent.parent.parent / 'ml'
sys.path.append(str(ml_path))

try:
    from working_raga_detector import raga_detector
except ImportError:
    # Fallback to creating a simple mock detector
    raga_detector = None

logger = logging.getLogger(__name__)

class RagaDetectionService:
    """
    Service wrapper for the working raga detection system.
    Provides a clean API for the FastAPI endpoints.
    """
    
    def __init__(self):
        """Initialize the raga detection service."""
        self.detector = raga_detector
        if self.detector is None:
            logger.warning("Working raga detector not available. Using mock service.")
            self.detector = MockRagaDetector()
    
    def predict(self, audio_path: str) -> Dict:
        """
        Predict the raga for the given audio file.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            dict: Prediction results
        """
        try:
            if self.detector is None:
                raise RuntimeError("Raga detector not initialized")
            
            # Get prediction from the working detector
            result = self.detector.predict(audio_path)
            
            # Add service metadata
            result['service_info'] = {
                'service_name': 'RagaDetectionService',
                'version': '1.0.0',
                'timestamp': datetime.utcnow().isoformat(),
                'processing_time': '~2 seconds'
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'supported_ragas': self.get_supported_ragas(),
                'service_info': {
                    'service_name': 'RagaDetectionService',
                    'version': '1.0.0',
                    'timestamp': datetime.utcnow().isoformat(),
                    'error': True
                }
            }
    
    def get_supported_ragas(self) -> list:
        """Get list of supported ragas."""
        try:
            if self.detector:
                return self.detector.get_supported_ragas()
            else:
                return ['Yaman', 'Bhairav', 'Kafi']  # Default supported ragas
        except Exception as e:
            logger.error(f"Error getting supported ragas: {e}")
            return ['Yaman', 'Bhairav', 'Kafi']
    
    def get_model_info(self) -> Dict:
        """Get information about the current model."""
        try:
            if self.detector:
                return self.detector.get_model_info()
            else:
                return {
                    'model_type': 'Mock',
                    'supported_ragas': ['Yaman', 'Bhairav', 'Kafi'],
                    'feature_count': 0,
                    'training_status': 'Mock',
                    'model_path': 'None'
                }
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {
                'model_type': 'Error',
                'supported_ragas': ['Yaman', 'Bhairav', 'Kafi'],
                'feature_count': 0,
                'training_status': 'Error',
                'model_path': 'None'
            }
    
    def is_ready(self) -> bool:
        """Check if the service is ready to handle requests."""
        try:
            return self.detector is not None and hasattr(self.detector, 'model')
        except Exception:
            return False

class MockRagaDetector:
    """
    Mock raga detector for testing when the real detector is not available.
    """
    
    def __init__(self):
        self.supported_ragas = ['Yaman', 'Bhairav', 'Kafi']
    
    def predict(self, audio_path: str) -> Dict:
        """Mock prediction that always returns a valid result."""
        import random
        
        # Simulate processing time
        import time
        time.sleep(0.1)
        
        # Random prediction for testing
        predicted_raga = random.choice(self.supported_ragas)
        confidence = random.uniform(0.6, 0.9)
        
        # Create mock top predictions
        top_predictions = []
        for raga in self.supported_ragas:
            prob = random.uniform(0.1, 0.8) if raga == predicted_raga else random.uniform(0.05, 0.3)
            top_predictions.append({
                'raga': raga,
                'probability': prob,
                'confidence': 'High' if prob > 0.7 else 'Medium' if prob > 0.4 else 'Low'
            })
        
        # Sort by probability
        top_predictions.sort(key=lambda x: x['probability'], reverse=True)
        
        return {
            'success': True,
            'predicted_raga': predicted_raga,
            'confidence': confidence,
            'top_predictions': top_predictions[:3],
            'supported_ragas': self.supported_ragas,
            'metadata': {
                'model_type': 'Mock',
                'feature_count': 0,
                'processing_time': '~0.1 seconds (mock)'
            }
        }
    
    def get_supported_ragas(self) -> list:
        return self.supported_ragas.copy()
    
    def get_model_info(self) -> Dict:
        return {
            'model_type': 'Mock',
            'supported_ragas': self.supported_ragas,
            'feature_count': 0,
            'training_status': 'Mock',
            'model_path': 'None'
        }

# Global service instance
raga_detection_service = RagaDetectionService()

# Backward compatibility - keep the old classifier interface
classifier = raga_detection_service
