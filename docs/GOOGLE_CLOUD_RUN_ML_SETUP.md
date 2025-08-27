# 🚀 **Google Cloud Run ML Model Hosting Setup**

## ✅ **Perfect Choice for ML Model Hosting!**

Your **Google Cloud Platform project "ragasense"** is perfect for hosting ML models! Google Cloud Run provides:

- ✅ **Serverless ML hosting** - No server management
- ✅ **Auto-scaling** - Handles traffic spikes automatically  
- ✅ **Container-based** - Easy ML model deployment
- ✅ **Cost-effective** - Pay only for actual usage
- ✅ **Global deployment** - Deploy close to your users
- ✅ **Perfect integration** - Works great with Convex

## 🏗️ **Architecture Overview**

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   React Frontend│    │   Convex Backend │    │  Google Cloud   │
│   (Vite + TS)   │◄──►│   (Database +    │◄──►│   Run ML API    │
│                 │    │    Auth + Files) │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                       │
                                              ┌─────────────────┐
                                              │  Google Cloud   │
                                              │   Storage       │
                                              │  (ML Models)    │
                                              └─────────────────┘
```

## 🔧 **Setup Steps**

### **1. Install Google Cloud CLI**

```bash
# macOS (using Homebrew)
brew install google-cloud-sdk

# Or download from: https://cloud.google.com/sdk/docs/install
```

### **2. Authenticate and Set Project**

```bash
# Login to Google Cloud
gcloud auth login

# Set your project
gcloud config set project ragasense

# Verify project
gcloud config get-value project
```

### **3. Enable Required APIs**

```bash
# Enable Cloud Run API
gcloud services enable run.googleapis.com

# Enable Cloud Storage API  
gcloud services enable storage.googleapis.com

# Enable Container Registry API
gcloud services enable containerregistry.googleapis.com
```

### **4. Setup ML Model Hosting**

```bash
# Run the setup script
python ml/cloud_run_setup.py

# This creates:
# - Dockerfile
# - FastAPI app (ml/cloud_run_app.py)
# - Requirements file
# - Deployment script
# - Convex integration
```

### **5. Upload Models to Cloud Storage**

```bash
# Upload ML models to GCS
python scripts/upload_models_to_gcs.py

# This will:
# - Create GCS bucket: gs://ragasense-models
# - Upload model files
# - Create metadata
```

### **6. Deploy to Cloud Run**

```bash
# Deploy the ML API
./deploy_to_cloud_run.sh

# This will:
# - Build Docker container
# - Deploy to Cloud Run
# - Set environment variables
# - Configure auto-scaling
```

## 📁 **File Structure Created**

```
project/
├── Dockerfile                    # Container configuration
├── ml/
│   └── cloud_run_app.py         # FastAPI ML API
├── requirements_cloud_run.txt    # Python dependencies
├── deploy_to_cloud_run.sh       # Deployment script
├── convex/
│   └── ml_integration.ts        # Convex integration
└── scripts/
    └── upload_models_to_gcs.py  # Model upload script
```

## 🎯 **ML API Endpoints**

### **Health Check**
```bash
curl https://raga-detection-api-ragasense.run.app/health
```

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": true,
  "service": "raga-detection-api"
}
```

### **Raga Detection**
```bash
curl -X POST https://raga-detection-api-ragasense.run.app/detect \
  -F "audio_file=@sample_raga.wav"
```

**Response:**
```json
{
  "predictions": [
    {
      "raga": "Yaman",
      "confidence": 0.85,
      "tradition": "Hindustani",
      "description": "Beautiful Yaman raga",
      "arohana": ["Sa", "Re", "Ga", "Ma", "Pa", "Dha", "Ni", "Sa"],
      "avarohana": ["Sa", "Ni", "Dha", "Pa", "Ma", "Ga", "Re", "Sa"]
    }
  ],
  "processing_time": 1.23,
  "audio_duration": 30.5,
  "sample_rate": 22050
}
```

### **Model Status**
```bash
curl https://raga-detection-api-ragasense.run.app/models/status
```

## 🔗 **Convex Integration**

The ML API integrates seamlessly with Convex:

```typescript
// In convex/ml_integration.ts
export const detectRaga = action({
  args: { audioFileId: v.id("_storage") },
  handler: async (ctx, args) => {
    // Get audio file from Convex Storage
    const audioUrl = await ctx.storage.getUrl(args.audioFileId);
    
    // Call Cloud Run ML API
    const response = await fetch(process.env.CLOUD_RUN_ML_URL + "/detect", {
      method: "POST",
      body: JSON.stringify({ audio_url: audioUrl }),
    });
    
    // Store results in Convex
    const result = await response.json();
    return await ctx.runMutation(api.ragaDetections.create, {
      predictions: result.predictions,
      confidence: result.predictions[0]?.confidence || 0,
    });
  },
});
```

## 💰 **Cost Estimation**

### **Cloud Run Pricing:**
- **CPU**: $0.00002400 per vCPU-second
- **Memory**: $0.00000250 per GiB-second
- **Requests**: $0.40 per million requests

### **Example Monthly Cost:**
- **1000 requests/day**: ~$12/month
- **10,000 requests/day**: ~$120/month
- **100,000 requests/day**: ~$1,200/month

### **Cloud Storage Pricing:**
- **ML Models**: ~60MB = ~$0.001/month
- **Audio files**: ~1GB = ~$0.02/month

**Total**: Very cost-effective for ML hosting!

## 🚀 **Deployment Commands**

### **Quick Deploy:**
```bash
# One-command deployment
./deploy_to_cloud_run.sh
```

### **Manual Deploy:**
```bash
# Build and deploy
gcloud run deploy raga-detection-api \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300 \
  --max-instances 10
```

### **Monitor Deployment:**
```bash
# View service details
gcloud run services describe raga-detection-api

# View logs
gcloud logs read --service=raga-detection-api

# List services
gcloud run services list
```

## 🔧 **Environment Variables**

Set these in Cloud Run:

```bash
MODEL_BUCKET=ragasense-models
CLOUD_RUN_ML_URL=https://raga-detection-api-ragasense.run.app
```

## 📊 **Monitoring & Logging**

### **View Logs:**
```bash
# Real-time logs
gcloud logs tail --service=raga-detection-api

# Recent logs
gcloud logs read --service=raga-detection-api --limit=50
```

### **Metrics Dashboard:**
- Go to [Google Cloud Console](https://console.cloud.google.com/run?authuser=2&hl=en&project=ragasense)
- Navigate to Cloud Run
- View metrics, logs, and performance

## 🧪 **Testing**

### **Test Health Endpoint:**
```bash
curl https://raga-detection-api-ragasense.run.app/health
```

### **Test Raga Detection:**
```bash
# Upload audio file
curl -X POST https://raga-detection-api-ragasense.run.app/detect \
  -F "audio_file=@tests/fixtures/test_raga_yaman.wav"
```

### **Test from Frontend:**
```typescript
// In your React app
const detectRaga = async (audioFile: File) => {
  const formData = new FormData();
  formData.append('audio_file', audioFile);
  
  const response = await fetch('https://raga-detection-api-ragasense.run.app/detect', {
    method: 'POST',
    body: formData,
  });
  
  return await response.json();
};
```

## 🔄 **CI/CD Pipeline**

### **GitHub Actions Workflow:**
```yaml
name: Deploy ML API
on:
  push:
    branches: [main]
    paths: ['ml/**']

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: google-github-actions/setup-gcloud@v0
      - run: |
          gcloud auth configure-docker
          gcloud run deploy raga-detection-api --source .
```

## 🎉 **Benefits Achieved**

### **1. Scalability:**
- ✅ **Auto-scaling**: 0 to 1000+ instances automatically
- ✅ **Global**: Deploy in multiple regions
- ✅ **Load balancing**: Automatic traffic distribution

### **2. Cost Efficiency:**
- ✅ **Pay-per-use**: Only pay for actual requests
- ✅ **No idle costs**: Scales to zero when not in use
- ✅ **Predictable pricing**: Clear per-request costs

### **3. Developer Experience:**
- ✅ **Simple deployment**: One command deployment
- ✅ **Easy monitoring**: Built-in logging and metrics
- ✅ **Version management**: Easy rollbacks and updates

### **4. Integration:**
- ✅ **Convex integration**: Seamless backend integration
- ✅ **REST API**: Standard HTTP endpoints
- ✅ **CORS support**: Frontend integration ready

## 🔗 **Useful Links**

- **Google Cloud Console**: https://console.cloud.google.com/run?authuser=2&hl=en&project=ragasense
- **Cloud Run Documentation**: https://cloud.google.com/run/docs
- **Cloud Storage**: https://console.cloud.google.com/storage/browser
- **Cloud Logging**: https://console.cloud.google.com/logs

## 🎯 **Next Steps**

1. **Deploy ML API**: `./deploy_to_cloud_run.sh`
2. **Test endpoints**: Use the curl commands above
3. **Integrate with Convex**: Update environment variables
4. **Monitor performance**: Check Cloud Console metrics
5. **Scale as needed**: Adjust memory/CPU settings

**Your ML model hosting is now ready for production! 🚀✨**
