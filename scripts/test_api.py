#!/usr/bin/env python3
"""
API Test Script

This script provides a simple way to test the Raga Detection API endpoints.
"""
import os
import sys
import json
import requests
from pathlib import Path

# Configuration
BASE_URL = "http://localhost:5000/api/v1"
TEST_AUDIO_FILE = "test_audio.wav"  # Placeholder for test audio file

def print_response(response):
    """Print the response in a formatted way."""
    print(f"Status Code: {response.status_code}")
    try:
        print("Response:")
        print(json.dumps(response.json(), indent=2))
    except ValueError:
        print("Response:", response.text)

def test_get_ragas():
    """Test the GET /ragas endpoint."""
    print("\n=== Testing GET /ragas ===")
    response = requests.get(f"{BASE_URL}/ragas")
    print_response(response)
    return response.ok

def test_get_raga(raga_id=1):
    """Test the GET /ragas/:id endpoint."""
    print(f"\n=== Testing GET /ragas/{raga_id} ===")
    response = requests.get(f"{BASE_URL}/ragas/{raga_id}")
    print_response(response)
    return response.ok

def test_search_ragas(query="yaman"):
    """Test the GET /ragas/search endpoint."""
    print(f"\n=== Testing GET /ragas/search?q={query} ===")
    response = requests.get(f"{BASE_URL}/ragas/search", params={"q": query})
    print_response(response)
    return response.ok

def test_analyze_audio(file_path=None):
    """Test the POST /analyze endpoint."""
    print("\n=== Testing POST /analyze ===")
    
    if not file_path or not os.path.exists(file_path):
        print(f"Test audio file not found: {file_path}")
        print("Please provide a valid audio file path for testing.")
        return False
    
    try:
        with open(file_path, 'rb') as f:
            files = {'file': (os.path.basename(file_path), f, 'audio/wav')}
            response = requests.post(f"{BASE_URL}/analyze", files=files)
        
        print_response(response)
        return response.ok
    except Exception as e:
        print(f"Error during file upload: {e}")
        return False

def run_tests():
    """Run all API tests."""
    print("=== Starting API Tests ===\n")
    
    tests = [
        ("Get All Ragas", test_get_ragas),
        ("Get Single Raga", lambda: test_get_raga(1)),
        ("Search Ragas", lambda: test_search_ragas("yaman")),
        ("Analyze Audio", lambda: test_analyze_audio(Path(__file__).parent.parent / "test_audio.wav"))
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n{'='*20} {name} {'='*20}")
        success = test_func()
        results.append((name, success))
    
    # Print summary
    print("\n=== Test Summary ===")
    for name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{status}: {name}")
    
    # Exit with appropriate status code
    if all(success for _, success in results):
        print("\nAll tests passed!")
        sys.exit(0)
    else:
        print("\nSome tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    run_tests()
