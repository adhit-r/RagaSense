#!/usr/bin/env python3
"""
Test script for the Raga Classifier ML model.
This script tests the model with sample audio data.
"""

import os
import sys
import numpy as np
import librosa
import tempfile
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ml.raga_classifier import RagaClassifier, create_simple_raga_classifier

def create_test_audio(duration: float = 5.0, sample_rate: int = 22050, frequency: float = 440.0):
    """Create a simple test audio signal."""
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    # Create a simple sine wave
    audio = np.sin(2 * np.pi * frequency * t)
    return audio, sample_rate

def test_model():
    """Test the raga classifier model."""
    print("Testing Raga Classifier Model...")
    
    try:
        # Create a simple classifier for testing
        classifier = create_simple_raga_classifier()
        print("✓ Simple classifier created successfully")
        
        # Create test audio
        audio, sr = create_test_audio(duration=10.0, frequency=440.0)
        print("✓ Test audio created successfully")
        
        # Save test audio to temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            import soundfile as sf
            sf.write(tmp_file.name, audio, sr)
            test_audio_path = tmp_file.name
        
        print("✓ Test audio saved to temporary file")
        
        # Test feature extraction
        features = classifier.extract_mel_features(test_audio_path)
        print(f"✓ Feature extraction successful. Feature shape: {features.shape}")
        
        # Test prediction
        result = classifier.predict_from_upload(test_audio_path)
        print("✓ Prediction successful")
        print(f"  Predicted raga: {result['predicted_raga']}")
        print(f"  Confidence: {result['confidence']:.3f}")
        print(f"  Top predictions: {result['top_predictions']}")
        
        # Clean up
        os.unlink(test_audio_path)
        print("✓ Test completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {str(e)}")
        return False

def test_model_with_real_audio(audio_path: str):
    """Test the model with a real audio file."""
    if not os.path.exists(audio_path):
        print(f"Audio file not found: {audio_path}")
        return False
    
    try:
        classifier = create_simple_raga_classifier()
        result = classifier.predict_from_upload(audio_path)
        
        print(f"✓ Prediction for {audio_path}:")
        print(f"  Predicted raga: {result['predicted_raga']}")
        print(f"  Confidence: {result['confidence']:.3f}")
        print(f"  Top predictions:")
        for pred in result['top_predictions']:
            print(f"    - {pred['raga']}: {pred['confidence']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {str(e)}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Test with provided audio file
        audio_file = sys.argv[1]
        test_model_with_real_audio(audio_file)
    else:
        # Run basic test
        test_model() 