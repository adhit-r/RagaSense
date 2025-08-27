#!/usr/bin/env python3
"""
Test script for the Raga Detection API.

This script sends a test audio file to the raga detection API and prints the results.
"""

import os
import sys
import requests
import json
from pathlib import Path

def test_raga_detection(api_url: str, audio_path: str):
    """
    Test the raga detection API with an audio file.
    
    Args:
        api_url: Base URL of the API (e.g., 'http://localhost:8000')
        audio_path: Path to the audio file to test
    """
    if not os.path.exists(audio_path):
        print(f"Error: File not found: {audio_path}")
        return
    
    url = f"{api_url}/api/ragas/detect"
    
    try:
        with open(audio_path, 'rb') as f:
            files = {'audio': (os.path.basename(audio_path), f, 'audio/wav')}
            response = requests.post(url, files=files)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("\nRaga Detection Results:")
            print("=" * 50)
            print(f"File: {os.path.basename(audio_path)}")
            print("-" * 50)
            
            for i, pred in enumerate(result['data']['predictions'], 1):
                print(f"{i}. {pred['raga']}: {pred['probability']:.2%}")
                print(f"   Aroha: {' '.join(pred['info'].get('aroha', ['N/A']))}")
                print(f"   Avaroha: {' '.join(pred['info'].get('avaroha', ['N/A']))}")
                print(f"   Time: {pred['info'].get('time', 'N/A')}")
                print(f"   Mood: {pred['info'].get('mood', 'N/A')}")
                print()
            
            print(f"\nMetadata: {json.dumps(result['metadata'], indent=2)}")
        else:
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"Error making request: {str(e)}")

def list_supported_ragas(api_url: str):
    """List all supported ragas."""
    try:
        response = requests.get(f"{api_url}/api/ragas/supported-ragas")
        if response.status_code == 200:
            data = response.json()
            print("\nSupported Ragas:")
            print("=" * 50)
            for i, raga in enumerate(data['data']['ragas'], 1):
                print(f"{i}. {raga}")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Raga Detection API')
    parser.add_argument('--url', default='http://localhost:8000', help='Base URL of the API')
    parser.add_argument('--list', action='store_true', help='List supported ragas')
    parser.add_argument('audio_file', nargs='?', help='Audio file to test (WAV, MP3, etc.)')
    
    args = parser.parse_args()
    
    if args.list:
        list_supported_ragas(args.url)
    elif args.audio_file:
        test_raga_detection(args.url, args.audio_file)
    else:
        parser.print_help()
        print("\nExamples:")
        print(f"  {sys.argv[0]} --list")
        print(f"  {sys.argv[0]} path/to/audio.wav")
        print(f"  {sys.argv[0]} --url http://api.example.com path/to/audio.wav")
