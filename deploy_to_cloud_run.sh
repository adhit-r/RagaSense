#!/bin/bash
# Deploy to Google Cloud Run

echo "🚀 Deploying Raga Detection API to Google Cloud Run..."

# Set project
gcloud config set project ragasense

# Enable required APIs
gcloud services enable run.googleapis.com
gcloud services enable storage.googleapis.com

# Create storage bucket for models
gsutil mb -l us-central1 gs://ragasense-models || echo "Bucket already exists"

# Build and deploy
gcloud run deploy raga-detection-api \
    --source . \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 2 \
    --timeout 300 \
    --max-instances 10 \
    --set-env-vars MODEL_BUCKET=ragasense-models

echo "✅ Deployment complete!"
echo "🌐 Service URL: https://raga-detection-api-ragasense.run.app"