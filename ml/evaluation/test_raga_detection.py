#!/usr/bin/env python3
"""
Test script for Raga Detection API
"""
import os
import requests
import json

def test_hugging_face_api():
    """Test the Hugging Face API directly"""
    print("🧪 Testing Hugging Face API...")
    
    api_key = os.getenv('HUGGINGFACE_API_KEY')
    if not api_key:
        print("❌ HUGGINGFACE_API_KEY not set")
        print("💡 Please set the environment variable:")
        print("   export HUGGINGFACE_API_KEY=your_actual_api_key")
        return False
    
    # Test the Carnatic Raga Classifier model
    model_url = "https://api-inference.huggingface.co/models/jeevster/carnatic-raga-classifier"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    try:
        # Test with a simple text input first
        response = requests.post(
            model_url,
            headers=headers,
            json={"inputs": "test audio data"}
        )
        
        print(f"✅ API Response Status: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ Hugging Face API is accessible")
            return True
        else:
            print(f"❌ API Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ API Test Failed: {e}")
        return False

def test_convex_connection():
    """Test Convex connection"""
    print("\n🧪 Testing Convex Connection...")
    
    try:
        # Test the Convex URL
        convex_url = "https://scrupulous-mosquito-279.convex.cloud"
        response = requests.get(f"{convex_url}/api/health")
        
        print(f"✅ Convex Status: {response.status_code}")
        return True
        
    except Exception as e:
        print(f"❌ Convex Test Failed: {e}")
        return False

def main():
    print("🎵 RAGA DETECTION TEST SUITE")
    print("=" * 50)
    
    # Test 1: Hugging Face API
    hf_working = test_hugging_face_api()
    
    # Test 2: Convex Connection
    convex_working = test_convex_connection()
    
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS")
    print("=" * 50)
    print(f"✅ Hugging Face API: {'Working' if hf_working else 'Failed'}")
    print(f"✅ Convex Connection: {'Working' if convex_working else 'Failed'}")
    
    if hf_working and convex_working:
        print("\n🎉 All tests passed! Raga detection should work.")
        print("\n🚀 Next steps:")
        print("1. Open your app at http://localhost:3000")
        print("2. Go to the Raga Detection page")
        print("3. Upload the test_audio.mp3 file")
        print("4. Test the raga detection feature")
    else:
        print("\n❌ Some tests failed. Check the errors above.")
    
    print("\n📝 Note: The first real API call may take 10-30 seconds")
    print("   as the model loads. Demo mode will work immediately.")

if __name__ == "__main__":
    main()
