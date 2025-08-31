#!/usr/bin/env python3
"""
Test script to check Hugging Face API connectivity
"""

import requests
import json
import base64
import os

def test_huggingface_api():
    """Test the Hugging Face Carnatic Raga Classifier API"""
    
    # API endpoint
    url = "https://api-inference.huggingface.co/models/jeevster/carnatic-raga-classifier"
    
    # Get API key from environment variable
    api_key = os.getenv("HUGGINGFACE_API_KEY", "your_api_key_here")
    
    if api_key == "your_api_key_here":
        print("⚠️ Please set HUGGINGFACE_API_KEY environment variable")
        print("export HUGGINGFACE_API_KEY=your_actual_api_key")
        return
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Create a simple test audio (just a placeholder)
    # In real usage, this would be actual audio data
    test_audio_base64 = "SGVsbG8gV29ybGQ="  # "Hello World" in base64
    
    payload = {
        "inputs": test_audio_base64
    }
    
    try:
        print("🧪 Testing Hugging Face API...")
        print(f"URL: {url}")
        print(f"API Key: {api_key[:10]}...")
        
        response = requests.post(url, headers=headers, json=payload)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            print("✅ API is working!")
            print(f"Response: {response.text[:200]}...")
        elif response.status_code == 404:
            print("⚠️ Model is still loading (404)")
            print("This is normal for the first few requests")
        elif response.status_code == 503:
            print("⚠️ Model is currently unavailable (503)")
            print("Please try again later")
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error testing API: {e}")

def test_model_status():
    """Test if the model is available"""
    
    url = "https://api-inference.huggingface.co/models/jeevster/carnatic-raga-classifier"
    api_key = os.getenv("HUGGINGFACE_API_KEY", "your_api_key_here")
    
    if api_key == "your_api_key_here":
        print("⚠️ Please set HUGGINGFACE_API_KEY environment variable")
        return
    
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    
    try:
        print("\n🔍 Checking model status...")
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            print("✅ Model is available!")
            model_info = response.json()
            print(f"Model: {model_info.get('modelId', 'Unknown')}")
            print(f"Pipeline: {model_info.get('pipeline_tag', 'Unknown')}")
        else:
            print(f"❌ Model status check failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error checking model status: {e}")

if __name__ == "__main__":
    print("🎵 Hugging Face API Test for RagaSense")
    print("=" * 50)
    
    test_huggingface_api()
    test_model_status()
    
    print("\n" + "=" * 50)
    print("💡 If you see 404/503 errors, the model might be loading.")
    print("💡 This is normal for Hugging Face free tier models.")
    print("💡 Try again in a few minutes.")



