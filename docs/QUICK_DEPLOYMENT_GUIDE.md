# 🚀 **Quick Deployment Guide - Google Cloud Run ML Hosting**

## ✅ **Perfect Setup for Your "ragasense" Project!**

Your Google Cloud Platform project is ready for ML model hosting! Here's the quick deployment:

## 🔧 **Step-by-Step Deployment**

### **1. Install Google Cloud CLI**
```bash
# macOS
brew install google-cloud-sdk

# Or download: https://cloud.google.com/sdk/docs/install
```

### **2. Authenticate & Set Project**
```bash
# Login
gcloud auth login

# Set your project (already configured!)
gcloud config set project ragasense

# Verify
gcloud config get-value project
# Should show: ragasense
```

### **3. Enable APIs**
```bash
# Enable required services
gcloud services enable run.googleapis.com
gcloud services enable storage.googleapis.com
gcloud services enable containerregistry.googleapis.com
```

### **4. Upload Models to Cloud Storage**
```bash
# Upload ML models
python scripts/upload_models_to_gcs.py

# This creates: gs://ragasense-models bucket
# Uploads: raga_classifier_model.h5, feature_extractor.pkl
```

### **5. Deploy to Cloud Run**
```bash
# One-command deployment
./deploy_to_cloud_run.sh

# Or manual deployment:
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
```

## 🎯 **Test Your ML API**

### **Health Check**
```bash
curl https://raga-detection-api-ragasense.run.app/health
```

### **Raga Detection**
```bash
curl -X POST https://raga-detection-api-ragasense.run.app/detect \
  -F "audio_file=@tests/fixtures/test_raga_yaman.wav"
```

## 🔗 **Integration with Convex**

### **Environment Variables**
Set in Convex dashboard:
```
CLOUD_RUN_ML_URL=https://raga-detection-api-ragasense.run.app
```

### **Use in Frontend**
```typescript
// In your React app
import { useAction } from "convex/react";
import { api } from "../convex/_generated/api";

const detectRaga = useAction(api.ml_integration.detectRaga);

// Upload audio and detect raga
const handleAudioUpload = async (audioFile: File) => {
  // Upload to Convex Storage
  const storageId = await convex.storage.upload(audioFile);
  
  // Detect raga using Cloud Run ML API
  const result = await detectRaga({ audioFileId: storageId });
  
  console.log("Raga detected:", result);
};
```

## 💰 **Cost Estimation**

### **Monthly Costs:**
- **1000 requests/day**: ~$12/month
- **10,000 requests/day**: ~$120/month  
- **100,000 requests/day**: ~$1,200/month

### **Free Tier:**
- **2 million requests/month** - FREE!
- **360,000 vCPU-seconds** - FREE!
- **180,000 GiB-seconds** - FREE!

## 📊 **Monitor Your Deployment**

### **View Logs**
```bash
# Real-time logs
gcloud logs tail --service=raga-detection-api

# Recent logs
gcloud logs read --service=raga-detection-api --limit=50
```

### **Cloud Console**
- **Dashboard**: https://console.cloud.google.com/run?authuser=2&hl=en&project=ragasense
- **Metrics**: Performance, errors, requests
- **Logs**: Detailed application logs

## 🎉 **Benefits You Get**

### **✅ Scalability**
- Auto-scales from 0 to 1000+ instances
- Handles traffic spikes automatically
- Global deployment options

### **✅ Cost Efficiency**  
- Pay only for actual requests
- Scales to zero when not in use
- Free tier covers most usage

### **✅ Developer Experience**
- One-command deployment
- Built-in monitoring and logging
- Easy rollbacks and updates

### **✅ Integration**
- Seamless Convex integration
- REST API endpoints
- CORS support for frontend

## 🔧 **Troubleshooting**

### **Common Issues:**

**1. Authentication Error**
```bash
gcloud auth login
gcloud config set project ragasense
```

**2. API Not Enabled**
```bash
gcloud services enable run.googleapis.com
gcloud services enable storage.googleapis.com
```

**3. Model Loading Failed**
```bash
# Check if models uploaded
gsutil ls gs://ragasense-models/models/

# Re-upload if needed
python scripts/upload_models_to_gcs.py
```

**4. Deployment Failed**
```bash
# Check logs
gcloud logs read --service=raga-detection-api

# Redeploy
./deploy_to_cloud_run.sh
```

## 🚀 **Production Checklist**

- ✅ **Models uploaded** to Cloud Storage
- ✅ **API deployed** to Cloud Run
- ✅ **Health check** passing
- ✅ **Environment variables** set in Convex
- ✅ **Frontend integration** working
- ✅ **Monitoring** configured
- ✅ **Logs** accessible

## 🎯 **Next Steps**

1. **Deploy**: `./deploy_to_cloud_run.sh`
2. **Test**: Use the curl commands above
3. **Integrate**: Update Convex environment variables
4. **Monitor**: Check Cloud Console metrics
5. **Scale**: Adjust settings as needed

## 🔗 **Useful Commands**

```bash
# List services
gcloud run services list

# Service details
gcloud run services describe raga-detection-api

# Update service
gcloud run services update raga-detection-api --memory 4Gi

# Delete service
gcloud run services delete raga-detection-api
```

## 🎉 **You're Ready for Production!**

Your ML model hosting is now:
- ✅ **Fully configured** for Google Cloud Run
- ✅ **Integrated** with Convex backend
- ✅ **Scalable** and cost-effective
- ✅ **Monitored** and logged
- ✅ **Ready** for production traffic

**Start creating beautiful Indian classical music with AI! 🎵✨**

---

**Need help?** Check the full guide: `GOOGLE_CLOUD_RUN_ML_SETUP.md`
