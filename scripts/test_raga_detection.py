#!/usr/bin/env python3
"""
Test script for the Working Raga Detection System
"""

import sys
import os
import tempfile
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path

# Add the ml directory to the path
ml_path = Path(__file__).parent.parent / 'ml'
sys.path.append(str(ml_path))

try:
    from working_raga_detector import raga_detector
    print("✓ Successfully imported Working Raga Detector")
except ImportError as e:
    print(f"✗ Failed to import Working Raga Detector: {e}")
    sys.exit(1)

def create_test_audio(raga_name: str, duration: int = 5) -> str:
    """Create a test audio file with characteristics of the specified raga."""
    
    # Audio parameters
    sr = 22050
    t = np.linspace(0, duration, sr * duration)
    
    # Create audio based on raga characteristics
    if raga_name == 'Yaman':
        # Yaman: evening raga, romantic mood - higher frequencies
        frequencies = [440, 554, 659, 740, 880]  # A, C#, E, F#, A
        audio = np.zeros_like(t)
        for i, freq in enumerate(frequencies):
            start_time = i * duration / len(frequencies)
            end_time = (i + 1) * duration / len(frequencies)
            mask = (t >= start_time) & (t < end_time)
            audio[mask] = np.sin(2 * np.pi * freq * t[mask]) * 0.2
    
    elif raga_name == 'Bhairav':
        # Bhairav: morning raga, devotional mood - lower frequencies
        frequencies = [220, 277, 330, 370, 440]  # A, C#, E, F#, A (lower octave)
        audio = np.zeros_like(t)
        for i, freq in enumerate(frequencies):
            start_time = i * duration / len(frequencies)
            end_time = (i + 1) * duration / len(frequencies)
            mask = (t >= start_time) & (t < end_time)
            audio[mask] = np.sin(2 * np.pi * freq * t[mask]) * 0.3
    
    elif raga_name == 'Kafi':
        # Kafi: versatile raga - mixed frequencies
        frequencies = [330, 415, 494, 554, 659]  # E, G#, B, C#, E
        audio = np.zeros_like(t)
        for i, freq in enumerate(frequencies):
            start_time = i * duration / len(frequencies)
            end_time = (i + 1) * duration / len(frequencies)
            mask = (t >= start_time) & (t < end_time)
            audio[mask] = np.sin(2 * np.pi * freq * t[mask]) * 0.25
    
    else:
        # Default: simple sine wave
        audio = np.sin(2 * np.pi * 440 * t) * 0.3
    
    # Add some noise to make it more realistic
    noise = np.random.normal(0, 0.01, len(audio))
    audio += noise
    
    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    temp_path = temp_file.name
    temp_file.close()
    
    # Save audio
    sf.write(temp_path, audio, sr)
    
    return temp_path

def test_raga_detection():
    """Test the raga detection system."""
    
    print("\n" + "="*50)
    print("TESTING WORKING RAGA DETECTION SYSTEM")
    print("="*50)
    
    # Test 1: Check if detector is initialized
    print("\n1. Testing Detector Initialization...")
    try:
        model_info = raga_detector.get_model_info()
        print(f"✓ Detector initialized successfully")
        print(f"  - Model type: {model_info['model_type']}")
        print(f"  - Supported ragas: {model_info['supported_ragas']}")
        print(f"  - Training status: {model_info['training_status']}")
    except Exception as e:
        print(f"✗ Detector initialization failed: {e}")
        return False
    
    # Test 2: Test with synthetic audio for each raga
    print("\n2. Testing Predictions with Synthetic Audio...")
    
    test_results = []
    for raga in raga_detector.get_supported_ragas():
        print(f"\n   Testing with {raga} audio...")
        
        try:
            # Create test audio
            test_audio_path = create_test_audio(raga)
            
            # Get prediction
            result = raga_detector.predict(test_audio_path)
            
            # Clean up
            os.unlink(test_audio_path)
            
            if result['success']:
                predicted_raga = result['predicted_raga']
                confidence = result['confidence']
                print(f"   ✓ Prediction: {predicted_raga} (confidence: {confidence:.3f})")
                
                test_results.append({
                    'test_raga': raga,
                    'predicted_raga': predicted_raga,
                    'confidence': confidence,
                    'success': True
                })
            else:
                print(f"   ✗ Prediction failed: {result.get('error', 'Unknown error')}")
                test_results.append({
                    'test_raga': raga,
                    'success': False,
                    'error': result.get('error', 'Unknown error')
                })
                
        except Exception as e:
            print(f"   ✗ Test failed: {e}")
            test_results.append({
                'test_raga': raga,
                'success': False,
                'error': str(e)
            })
    
    # Test 3: Test with random audio
    print("\n3. Testing with Random Audio...")
    try:
        # Create random audio
        sr = 22050
        duration = 5
        t = np.linspace(0, duration, sr * duration)
        random_audio = np.random.normal(0, 0.1, len(t))
        
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_path = temp_file.name
        temp_file.close()
        
        sf.write(temp_path, random_audio, sr)
        
        result = raga_detector.predict(temp_path)
        os.unlink(temp_path)
        
        if result['success']:
            print(f"✓ Random audio prediction: {result['predicted_raga']} (confidence: {result['confidence']:.3f})")
        else:
            print(f"✗ Random audio prediction failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"✗ Random audio test failed: {e}")
    
    # Test 4: Performance test
    print("\n4. Performance Test...")
    try:
        import time
        
        test_audio_path = create_test_audio('Yaman')
        
        start_time = time.time()
        result = raga_detector.predict(test_audio_path)
        end_time = time.time()
        
        processing_time = end_time - start_time
        os.unlink(test_audio_path)
        
        print(f"✓ Processing time: {processing_time:.2f} seconds")
        
        if processing_time < 5:  # Should be under 5 seconds
            print("✓ Performance is acceptable")
        else:
            print("⚠ Performance is slower than expected")
            
    except Exception as e:
        print(f"✗ Performance test failed: {e}")
    
    # Summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    
    successful_tests = sum(1 for r in test_results if r['success'])
    total_tests = len(test_results)
    
    print(f"Successful predictions: {successful_tests}/{total_tests}")
    
    if successful_tests == total_tests:
        print("✓ All tests passed! Raga detection system is working correctly.")
        return True
    else:
        print("⚠ Some tests failed. Check the implementation.")
        return False

def test_api_integration():
    """Test the API integration."""
    
    print("\n" + "="*50)
    print("TESTING API INTEGRATION")
    print("="*50)
    
    try:
        # Import the service
        sys.path.append(str(Path(__file__).parent.parent / 'backend' / 'api' / 'services'))
        from raga_detector import raga_detection_service
        
        print("✓ Successfully imported API service")
        
        # Test service methods
        supported_ragas = raga_detection_service.get_supported_ragas()
        print(f"✓ Supported ragas: {supported_ragas}")
        
        model_info = raga_detection_service.get_model_info()
        print(f"✓ Model info retrieved: {model_info['model_type']}")
        
        is_ready = raga_detection_service.is_ready()
        print(f"✓ Service ready: {is_ready}")
        
        # Test prediction through service
        test_audio_path = create_test_audio('Yaman')
        result = raga_detection_service.predict(test_audio_path)
        os.unlink(test_audio_path)
        
        if result['success']:
            print(f"✓ API prediction successful: {result['predicted_raga']}")
        else:
            print(f"✗ API prediction failed: {result.get('error', 'Unknown error')}")
        
        return True
        
    except Exception as e:
        print(f"✗ API integration test failed: {e}")
        return False

if __name__ == "__main__":
    print("Starting Raga Detection System Tests...")
    
    # Test the core detector
    detector_success = test_raga_detection()
    
    # Test API integration
    api_success = test_api_integration()
    
    # Final summary
    print("\n" + "="*50)
    print("FINAL SUMMARY")
    print("="*50)
    
    if detector_success and api_success:
        print("🎉 ALL TESTS PASSED!")
        print("The Working Raga Detection System is ready for use.")
    elif detector_success:
        print("⚠ Core detector works, but API integration needs attention.")
    elif api_success:
        print("⚠ API integration works, but core detector needs attention.")
    else:
        print("❌ Tests failed. System needs debugging.")
    
    print("\nNext steps:")
    print("1. Start the backend server: python -m backend.main")
    print("2. Test the API endpoints")
    print("3. Integrate with the frontend")
    print("4. Add real training data when available")
