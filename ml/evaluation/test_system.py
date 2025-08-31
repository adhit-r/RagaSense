#!/usr/bin/env python3
"""
Quick test script to verify RagaSense system is working
"""

import requests
import json
import tempfile
import os

def test_backend_health():
    """Test backend health endpoint"""
    try:
        response = requests.get("http://localhost:8002/health")
        if response.status_code == 200:
            data = response.json()
            print("✅ Backend Health Check:")
            print(f"   Status: {data['status']}")
            print(f"   Service: {data['service']}")
            print(f"   Ragas Loaded: {data['ragas_loaded']}")
            print(f"   Model Loaded: {data['model_loaded']}")
            return True
        else:
            print(f"❌ Backend health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Backend health check error: {e}")
        return False

def test_frontend():
    """Test frontend accessibility"""
    try:
        response = requests.get("http://localhost:3000")
        if response.status_code == 200:
            print("✅ Frontend is accessible")
            return True
        else:
            print(f"❌ Frontend check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Frontend check error: {e}")
        return False

def test_raga_detection():
    """Test raga detection with a dummy audio file"""
    try:
        # Create a dummy audio file for testing
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            # Write some dummy audio data (just zeros for testing)
            tmp_file.write(b'\x00' * 1024)  # 1KB of zeros
            tmp_file_path = tmp_file.name
        
        # Test the detection endpoint
        with open(tmp_file_path, 'rb') as f:
            files = {'file': ('test.wav', f, 'audio/wav')}
            response = requests.post("http://localhost:8002/api/detect-raga", files=files)
        
        # Clean up temp file
        os.unlink(tmp_file_path)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Raga Detection Test:")
            print(f"   Success: {data.get('success', False)}")
            if data.get('success') and data.get('prediction'):
                pred = data['prediction']
                print(f"   Detected Raga: {pred.get('raga', 'Unknown')}")
                print(f"   Confidence: {pred.get('confidence', 0):.2%}")
                print(f"   Style: {pred.get('style', 'Unknown')}")
                if pred.get('top_predictions'):
                    print(f"   Top Predictions: {len(pred['top_predictions'])}")
            return True
        else:
            print(f"❌ Raga detection test failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Raga detection test error: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing RagaSense System...\n")
    
    tests = [
        ("Backend Health", test_backend_health),
        ("Frontend Access", test_frontend),
        ("Raga Detection", test_raga_detection),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"Testing {test_name}...")
        result = test_func()
        results.append((test_name, result))
        print()
    
    # Summary
    print("📊 Test Results Summary:")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! RagaSense is ready to use!")
        print("\n📱 Next steps:")
        print("1. Visit http://localhost:3000 to use the enhanced UI")
        print("2. Try recording audio or uploading files")
        print("3. Test the raga detection with real music")
        print("4. Explore the professional B2C design features")
    else:
        print("\n⚠️ Some tests failed. Please check the logs above.")

if __name__ == "__main__":
    main()

