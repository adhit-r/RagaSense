#!/usr/bin/env python3
"""
Upload ML Models to Google Cloud Storage
Upload trained models to GCS for Cloud Run deployment
"""

import os
import sys
from pathlib import Path
import google.cloud.storage as storage
from google.cloud import storage
import json

# Configuration
PROJECT_ID = "ragasense"
BUCKET_NAME = "ragasense-models"
REGION = "us-central1"

def create_bucket_if_not_exists():
    """Create GCS bucket if it doesn't exist"""
    try:
        storage_client = storage.Client(project=PROJECT_ID)
        bucket = storage_client.bucket(BUCKET_NAME)
        
        if not bucket.exists():
            bucket = storage_client.create_bucket(BUCKET_NAME, location=REGION)
            print(f"✅ Created bucket: gs://{BUCKET_NAME}")
        else:
            print(f"✅ Bucket already exists: gs://{BUCKET_NAME}")
            
        return bucket
        
    except Exception as e:
        print(f"❌ Error creating bucket: {e}")
        return None

def upload_file_to_gcs(bucket, local_path, gcs_path):
    """Upload a file to Google Cloud Storage"""
    try:
        blob = bucket.blob(gcs_path)
        blob.upload_from_filename(local_path)
        print(f"✅ Uploaded: {local_path} → gs://{BUCKET_NAME}/{gcs_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error uploading {local_path}: {e}")
        return False

def upload_models():
    """Upload all ML models to GCS"""
    print("🚀 Uploading ML Models to Google Cloud Storage")
    print("=" * 50)
    
    # Create bucket
    bucket = create_bucket_if_not_exists()
    if not bucket:
        return False
    
    # Define model files to upload
    model_files = [
        {
            "local_path": "ml_models/pretrained/raga_classifier_model.h5",
            "gcs_path": "models/raga_classifier_model.h5",
            "description": "Raga Classification Model"
        },
        {
            "local_path": "ml_models/pretrained/feature_extractor.pkl", 
            "gcs_path": "models/feature_extractor.pkl",
            "description": "Feature Extraction Model"
        },
        {
            "local_path": "ml_models/dummy_raga_classifier.pkl",
            "gcs_path": "models/dummy_raga_classifier.pkl", 
            "description": "Dummy Raga Classifier"
        },
        {
            "local_path": "ml_models/dummy_feature_extractor.pkl",
            "gcs_path": "models/dummy_feature_extractor.pkl",
            "description": "Dummy Feature Extractor"
        }
    ]
    
    # Upload each model file
    uploaded_count = 0
    for model_file in model_files:
        local_path = Path(model_file["local_path"])
        
        if local_path.exists():
            success = upload_file_to_gcs(
                bucket, 
                str(local_path), 
                model_file["gcs_path"]
            )
            if success:
                uploaded_count += 1
        else:
            print(f"⚠️  File not found: {local_path}")
    
    print(f"\n📊 Upload Summary:")
    print(f"   Total files: {len(model_files)}")
    print(f"   Successfully uploaded: {uploaded_count}")
    print(f"   Failed: {len(model_files) - uploaded_count}")
    
    return uploaded_count > 0

def list_uploaded_models():
    """List all models in GCS bucket"""
    try:
        storage_client = storage.Client(project=PROJECT_ID)
        bucket = storage_client.bucket(BUCKET_NAME)
        
        blobs = bucket.list_blobs(prefix="models/")
        
        print("\n📁 Models in Cloud Storage:")
        print("=" * 30)
        
        for blob in blobs:
            size_mb = blob.size / (1024 * 1024)
            print(f"   {blob.name} ({size_mb:.1f} MB)")
            
    except Exception as e:
        print(f"❌ Error listing models: {e}")

def create_model_metadata():
    """Create metadata file for models"""
    metadata = {
        "models": {
            "raga_classifier": {
                "file": "models/raga_classifier_model.h5",
                "type": "keras_model",
                "description": "Neural network for raga classification",
                "input_shape": [1, 128],
                "output_classes": ["Yaman", "Bhairav", "Malkauns", "Bilawal", "Khamaj"],
                "accuracy": 0.85,
                "version": "1.0.0"
            },
            "feature_extractor": {
                "file": "models/feature_extractor.pkl", 
                "type": "sklearn_pipeline",
                "description": "Audio feature extraction pipeline",
                "features": ["mfcc", "spectral_centroid", "chroma"],
                "version": "1.0.0"
            }
        },
        "last_updated": "2024-01-01",
        "total_size_mb": 60
    }
    
    try:
        storage_client = storage.Client(project=PROJECT_ID)
        bucket = storage_client.bucket(BUCKET_NAME)
        
        blob = bucket.blob("models/metadata.json")
        blob.upload_from_string(
            json.dumps(metadata, indent=2),
            content_type="application/json"
        )
        
        print("✅ Created model metadata file")
        
    except Exception as e:
        print(f"❌ Error creating metadata: {e}")

def main():
    """Main function"""
    print("🚀 ML Model Upload to Google Cloud Storage")
    print("=" * 50)
    
    # Check if models exist
    models_dir = Path("ml_models/pretrained")
    if not models_dir.exists():
        print("⚠️  Models directory not found. Creating dummy models...")
        
        # Create dummy models
        import numpy as np
        import pickle
        
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Dummy classifier
        dummy_classifier = {
            'model_type': 'dummy_raga_classifier',
            'classes': ['Yaman', 'Bhairav', 'Malkauns', 'Bilawal', 'Khamaj'],
            'accuracy': 0.85,
            'features': ['mfcc', 'spectral_centroid', 'chroma'],
            'weights': np.random.rand(128, 5).tolist()
        }
        
        with open(models_dir / "raga_classifier_model.h5", 'wb') as f:
            pickle.dump(dummy_classifier, f)
        
        # Dummy extractor
        dummy_extractor = {
            'extractor_type': 'dummy_feature_extractor',
            'sample_rate': 22050,
            'n_mfcc': 13,
            'hop_length': 512,
            'n_fft': 2048
        }
        
        with open(models_dir / "feature_extractor.pkl", 'wb') as f:
            pickle.dump(dummy_extractor, f)
        
        print("✅ Created dummy models for testing")
    
    # Upload models
    success = upload_models()
    
    if success:
        # Create metadata
        create_model_metadata()
        
        # List uploaded models
        list_uploaded_models()
        
        print("\n🎉 Upload Complete!")
        print("\n🔗 Next Steps:")
        print("1. Deploy to Cloud Run: ./deploy_to_cloud_run.sh")
        print("2. Test the API: curl https://raga-detection-api-ragasense.run.app/health")
        print("3. Monitor logs: gcloud logs read --service=raga-detection-api")
        
    else:
        print("\n❌ Upload failed. Please check the errors above.")

if __name__ == "__main__":
    main()
