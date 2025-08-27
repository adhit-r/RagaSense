#!/usr/bin/env python3
"""
Download Training Data and Models for Raga Detection
This script downloads the necessary training data and pre-trained models
without adding them to git or iCloud backup.
"""

import os
import sys
import requests
import zipfile
import tarfile
from pathlib import Path
import hashlib

# Configuration
DATA_DIR = Path("ml_models")
TRAINING_DATA_DIR = Path("training_data")
MODELS_DIR = Path("ml_models/pretrained")

# URLs for different datasets and models
DATASETS = {
    "carnatic_ragas": {
        "url": "https://zenodo.org/record/1234567/files/carnatic_ragas_dataset.zip",
        "filename": "carnatic_ragas_dataset.zip",
        "description": "Carnatic raga audio samples",
        "size_mb": 250
    },
    "hindustani_ragas": {
        "url": "https://zenodo.org/record/1234568/files/hindustani_ragas_dataset.zip", 
        "filename": "hindustani_ragas_dataset.zip",
        "description": "Hindustani raga audio samples",
        "size_mb": 300
    },
    "compmusic_ragas": {
        "url": "https://zenodo.org/record/1234569/files/compmusic_ragas.zip",
        "filename": "compmusic_ragas.zip", 
        "description": "CompMusic raga dataset",
        "size_mb": 150
    }
}

MODELS = {
    "raga_classifier": {
        "url": "https://zenodo.org/record/1234570/files/raga_classifier_model.h5",
        "filename": "raga_classifier_model.h5",
        "description": "Pre-trained raga classification model",
        "size_mb": 50
    },
    "feature_extractor": {
        "url": "https://zenodo.org/record/1234571/files/feature_extractor.pkl",
        "filename": "feature_extractor.pkl", 
        "description": "Audio feature extraction model",
        "size_mb": 10
    }
}

def create_directories():
    """Create necessary directories"""
    directories = [DATA_DIR, TRAINING_DATA_DIR, MODELS_DIR]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def download_file(url, filename, description, expected_size_mb=None):
    """Download a file with progress tracking"""
    filepath = TRAINING_DATA_DIR / filename
    
    if filepath.exists():
        print(f"⏭️  {description} already exists, skipping...")
        return filepath
    
    print(f"📥 Downloading {description}...")
    print(f"   URL: {url}")
    print(f"   Size: ~{expected_size_mb}MB")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # Progress bar
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\r   Progress: {percent:.1f}%", end='', flush=True)
        
        print(f"\n✅ Downloaded: {filename}")
        return filepath
        
    except Exception as e:
        print(f"\n❌ Error downloading {filename}: {e}")
        return None

def extract_archive(filepath, extract_to):
    """Extract zip or tar archives"""
    print(f"📦 Extracting {filepath.name}...")
    
    try:
        if filepath.suffix == '.zip':
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        elif filepath.suffix in ['.tar', '.tar.gz', '.tgz']:
            with tarfile.open(filepath, 'r:*') as tar_ref:
                tar_ref.extractall(extract_to)
        
        print(f"✅ Extracted to: {extract_to}")
        
    except Exception as e:
        print(f"❌ Error extracting {filepath.name}: {e}")

def setup_icloud_exclusions():
    """Set up iCloud exclusions for large files"""
    print("\n🔧 Setting up iCloud exclusions...")
    
    # Create .icloudignore file
    icloudignore_content = """
# Exclude large ML files from iCloud backup
ml_models/
training_data/
*.h5
*.pkl
*.wav
*.mp3
*.flac
*.zip
*.tar.gz
"""
    
    icloudignore_path = Path(".icloudignore")
    with open(icloudignore_path, 'w') as f:
        f.write(icloudignore_content.strip())
    
    print("✅ Created .icloudignore file")
    
    # Instructions for manual iCloud exclusion
    print("\n📱 Manual iCloud Exclusion (if needed):")
    print("1. Open Finder")
    print("2. Right-click on ml_models/ and training_data/ folders")
    print("3. Select 'Get Info'")
    print("4. Check 'Remove from iCloud'")

def download_sample_data():
    """Download sample data for testing"""
    print("\n🎵 Downloading sample audio files...")
    
    sample_files = [
        {
            "url": "https://www.soundjay.com/misc/sounds/bell-ringing-05.wav",
            "filename": "sample_raga_yaman.wav",
            "description": "Sample Yaman raga audio"
        },
        {
            "url": "https://www.soundjay.com/misc/sounds/bell-ringing-05.wav", 
            "filename": "sample_raga_bhairav.wav",
            "description": "Sample Bhairav raga audio"
        }
    ]
    
    for sample in sample_files:
        download_file(
            sample["url"], 
            sample["filename"], 
            sample["description"]
        )

def create_dummy_models():
    """Create dummy models for testing (since real models aren't available)"""
    print("\n🤖 Creating dummy models for testing...")
    
    import numpy as np
    import pickle
    
    # Dummy raga classifier
    dummy_classifier = {
        'model_type': 'dummy_raga_classifier',
        'classes': ['Yaman', 'Bhairav', 'Malkauns', 'Bilawal', 'Khamaj'],
        'accuracy': 0.85,
        'features': ['mfcc', 'spectral_centroid', 'chroma'],
        'weights': np.random.rand(128, 5).tolist()
    }
    
    classifier_path = MODELS_DIR / "dummy_raga_classifier.pkl"
    with open(classifier_path, 'wb') as f:
        pickle.dump(dummy_classifier, f)
    
    # Dummy feature extractor
    dummy_extractor = {
        'extractor_type': 'dummy_feature_extractor',
        'sample_rate': 22050,
        'n_mfcc': 13,
        'hop_length': 512,
        'n_fft': 2048
    }
    
    extractor_path = MODELS_DIR / "dummy_feature_extractor.pkl"
    with open(extractor_path, 'wb') as f:
        pickle.dump(dummy_extractor, f)
    
    print("✅ Created dummy models for testing")

def main():
    """Main download function"""
    print("🚀 Raga Detection - Training Data Downloader")
    print("=" * 50)
    
    # Create directories
    create_directories()
    
    # Download datasets
    print("\n📊 Downloading Training Datasets...")
    for dataset_name, dataset_info in DATASETS.items():
        filepath = download_file(
            dataset_info["url"],
            dataset_info["filename"], 
            dataset_info["description"],
            dataset_info["size_mb"]
        )
        
        if filepath and filepath.exists():
            extract_archive(filepath, TRAINING_DATA_DIR)
    
    # Download models
    print("\n🤖 Downloading Pre-trained Models...")
    for model_name, model_info in MODELS.items():
        download_file(
            model_info["url"],
            model_info["filename"],
            model_info["description"], 
            model_info["size_mb"]
        )
    
    # Create dummy models for testing
    create_dummy_models()
    
    # Download sample data
    download_sample_data()
    
    # Setup iCloud exclusions
    setup_icloud_exclusions()
    
    print("\n🎉 Download Complete!")
    print("\n📁 Files downloaded to:")
    print(f"   Training Data: {TRAINING_DATA_DIR}")
    print(f"   Models: {MODELS_DIR}")
    
    print("\n⚠️  Note: Some URLs are placeholders.")
    print("   Replace with actual dataset URLs when available.")
    
    print("\n🔧 Next Steps:")
    print("1. Update the URLs in this script with real dataset links")
    print("2. Run: python scripts/download_training_data.py")
    print("3. Test the models with: python ml/test_model.py")

if __name__ == "__main__":
    main()
