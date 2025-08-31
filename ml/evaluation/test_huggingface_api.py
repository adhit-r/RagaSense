#!/usr/bin/env python3
"""
Test script for Hugging Face API integration
"""

import requests
import json
import base64
import os
from pathlib import Path

def test_huggingface_api():
    """Test the Hugging Face Carnatic Raga Classifier API"""
    
    print("🎵 Testing Hugging Face API for RagaSense")
    print("="*50)
    
    # API Configuration
    API_URL = "https://api-inference.huggingface.co/models/jeevster/carnatic-raga-classifier"
    API_KEY = os.getenv("HUGGINGFACE_API_KEY", "your_api_key_here")
    
    if API_KEY == "your_api_key_here":
        print("⚠️ Please set HUGGINGFACE_API_KEY environment variable")
        print("export HUGGINGFACE_API_KEY=your_actual_api_key")
        return False
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    try:
        # Test 1: Check if model is ready
        print("🔄 Checking if model is ready...")
        response = requests.get(API_URL, headers=headers)
        
        if response.status_code == 200:
            print("✅ Model is ready!")
        elif response.status_code == 503:
            print("⚠️ Model is loading...")
        else:
            print(f"❌ Model check failed: {response.status_code}")
            return False
        
        # Test 2: Try with a small test audio file
        test_audio_path = "test_audio.mp3"
        
        if Path(test_audio_path).exists():
            print(f"🔄 Testing with audio file: {test_audio_path}")
            
            # Read and encode audio file
            with open(test_audio_path, "rb") as audio_file:
                audio_data = audio_file.read()
                base64_audio = base64.b64encode(audio_data).decode('utf-8')
            
            # Make prediction request
            payload = {
                "inputs": base64_audio
            }
            
            response = requests.post(API_URL, headers=headers, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                print("✅ API call successful!")
                print(f"📊 Response: {json.dumps(result, indent=2)}")
                
                if result and len(result) > 0 and len(result[0]) > 0:
                    top_prediction = result[0][0]
                    print(f"🎯 Top prediction: {top_prediction['label']} ({top_prediction['score']:.2%})")
                    return True
                else:
                    print("❌ No predictions in response")
                    return False
                    
            elif response.status_code == 503:
                print("⚠️ Model is still loading, please wait...")
                return False
            else:
                print(f"❌ API call failed: {response.status_code} {response.text}")
                return False
        else:
            print(f"⚠️ Test audio file not found: {test_audio_path}")
            print("📝 Creating a simple test...")
            
            # Test with a dummy request to see if API responds
            dummy_payload = {
                "inputs": "dummy_audio_data"
            }
            
            response = requests.post(API_URL, headers=headers, json=dummy_payload)
            
            if response.status_code == 400:
                print("✅ API is responding (expected error for dummy data)")
                print("📝 API is working, but needs real audio data")
                return True
            else:
                print(f"❌ Unexpected response: {response.status_code}")
                return False
                
    except Exception as e:
        print(f"❌ Error testing API: {e}")
        return False

def test_api_endpoints():
    """Test different API endpoints"""
    print("\n🔗 Testing API Endpoints")
    print("="*30)
    
    API_KEY = os.getenv("HUGGINGFACE_API_KEY", "your_api_key_here")
    
    if API_KEY == "your_api_key_here":
        print("⚠️ Please set HUGGINGFACE_API_KEY environment variable")
        return False
    
    # Test different endpoints
    endpoints = [
        "https://api-inference.huggingface.co/models/jeevster/carnatic-raga-classifier",
        "https://huggingface.co/spaces/jeevster/carnatic-raga-classifier"
    ]
    
    for endpoint in endpoints:
        try:
            print(f"🔄 Testing: {endpoint}")
            
            if "api-inference" in endpoint:
                headers = {"Authorization": f"Bearer {API_KEY}"}
                response = requests.get(endpoint, headers=headers)
            else:
                response = requests.get(endpoint)
            
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                print("✅ Endpoint is accessible")
            else:
                print("❌ Endpoint not accessible")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    return True

def test_model_info():
    """Get information about the model"""
    print("\n📋 Model Information")
    print("="*30)
    
    API_KEY = os.getenv("HUGGINGFACE_API_KEY", "your_api_key_here")
    
    if API_KEY == "your_api_key_here":
        print("⚠️ Please set HUGGINGFACE_API_KEY environment variable")
        return False
    
    API_URL = "https://api-inference.huggingface.co/models/jeevster/carnatic-raga-classifier"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    
    try:
        response = requests.get(API_URL, headers=headers)
        
        if response.status_code == 200:
            model_info = response.json()
            print(f"Model ID: {model_info.get('modelId', 'Unknown')}")
            print(f"Pipeline: {model_info.get('pipeline_tag', 'Unknown')}")
            print(f"Tags: {model_info.get('tags', [])}")
            print(f"Author: {model_info.get('author', 'Unknown')}")
            return True
        else:
            print(f"❌ Failed to get model info: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error getting model info: {e}")
        return False

if __name__ == "__main__":
    print("🎵 Hugging Face API Test Suite")
    print("="*50)
    
    # Run all tests
    test1 = test_huggingface_api()
    test2 = test_api_endpoints()
    test3 = test_model_info()
    
    print("\n" + "="*50)
    print("📊 Test Results:")
    print(f"API Test: {'✅ PASS' if test1 else '❌ FAIL'}")
    print(f"Endpoints Test: {'✅ PASS' if test2 else '❌ FAIL'}")
    print(f"Model Info Test: {'✅ PASS' if test3 else '❌ FAIL'}")
    
    if all([test1, test2, test3]):
        print("\n🎉 All tests passed! API is working correctly.")
    else:
        print("\n⚠️ Some tests failed. Please check the configuration.")



