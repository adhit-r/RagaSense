# **RagaSense Documentation**

Welcome to the comprehensive documentation for **RagaSense** - an AI-powered Indian classical music platform with raga detection and music generation capabilities.

## **Project Overview**

RagaSense is a full-stack application that combines:
- **AI Raga Detection**: Upload audio files and identify Indian classical ragas
- **AI Music Generation**: Create personalized Indian classical music
- **Real-time Features**: Live updates and progress tracking
- **Modern Tech Stack**: Lynx, Convex, Google Cloud Run

## **Documentation Structure**

### **Getting Started**
- **[Quick Start Guide](QUICK_DEPLOYMENT_GUIDE.md)** - Get up and running in minutes
- **[Answers to Common Questions](ANSWERS_TO_YOUR_QUESTIONS.md)** - FAQ and troubleshooting

### **Architecture & Setup**
- **[Convex Implementation Complete](CONVEX_IMPLEMENTATION_COMPLETE.md)** - Complete backend migration to Convex
- **[Convex Migration Plan](CONVEX_MIGRATION_PLAN.md)** - Detailed migration strategy and architecture
- **[Google Cloud Run ML Setup](GOOGLE_CLOUD_RUN_ML_SETUP.md)** - ML model hosting with Google Cloud Run

### **Project Management**
- **[Organization Summary](ORGANIZATION_SUMMARY.md)** - Codebase organization and structure
- **[Lynx Consolidation Summary](LYNX_CONSOLIDATION_AND_AI_GENERATION_SUMMARY.md)** - Frontend framework consolidation
- **[AI Music Generation PRD](ai_music_generation_prd.md)** - Product requirements for AI music generation
- **[API Documentation](API_DOCS.md)** - Backend API endpoints and usage
- **[TODO List](TODO.md)** - Current tasks and future roadmap

## **Core Features**

### **1. AI Raga Detection**
- Upload audio files (WAV, MP3, FLAC, etc.)
- Real-time raga identification using ML models
- Detailed raga information and scale patterns
- Support for both Hindustani and Carnatic traditions

### **2. AI Music Generation**
- 5-step generation process:
  1. **Music Type Selection** (Instrumental/Vocal)
  2. **Voice/Instrument Selection** (Detailed options)
  3. **Mood Selection** (With smart raga suggestions)
  4. **Theme Selection** (Cultural context)
  5. **Generation Process** (Real-time progress)
- Customizable parameters and preferences
- Download and sharing capabilities

### **3. Real-time Features**
- Live generation progress updates
- Real-time database synchronization
- Instant UI updates
- Offline support with automatic sync

## **Technology Stack**

### **Frontend**
- **Lynx** - Cross-platform framework for web and mobile
- **TypeScript** for type safety
- **Tailwind CSS** for styling
- **Convex Client** for real-time backend integration

### **Backend**
- **Convex** (Database + Auth + Real-time + File Storage)
- **Google Cloud Run** (ML Model Hosting)
- **Google Cloud Storage** (Model Storage)

### **ML & AI**
- **TensorFlow/Keras** for neural networks
- **Librosa** for audio processing
- **Scikit-learn** for feature extraction
- **FastAPI** for ML API endpoints

### **Development Tools**
- **Bun** for package management
- **TypeScript** for type safety
- **ESLint** for code quality
- **Git** for version control

## **Architecture Overview**

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Lynx Frontend │    │   Convex Backend │    │  Google Cloud   │
│   (Web + Mobile)│◄──►│   (Database +    │◄──►│   Run ML API    │
│                 │    │    Auth + Files) │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                       │
                                              ┌─────────────────┐
                                              │  Google Cloud   │
                                              │   Storage       │
                                              │  (ML Models)    │
                                              └─────────────────┘
```

## **Quick Start**

### **1. Prerequisites**
- Node.js 18+ or Bun
- Python 3.9+
- Google Cloud CLI (for ML hosting)

### **2. Installation**
```bash
# Clone repository
git clone <repository-url>
cd raga_detector

# Install dependencies
cd frontend && bun install
cd .. && pip install -r requirements.txt
```

### **3. Development**
```bash
# Start frontend
cd frontend && bun run dev

# Start Convex
bunx convex dev

# Start ML API (optional)
python ml/cloud_run_app.py
```

### **4. Production Deployment**
```bash
# Deploy ML models
python scripts/upload_models_to_gcs.py
./deploy_to_cloud_run.sh

# Deploy frontend
cd frontend && bun run build
```

## **Project Structure**

```
raga_detector/
├── frontend/                 # Lynx frontend application
│   ├── src/
│   │   ├── components/      # Reusable UI components
│   │   ├── pages/          # Page components
│   │   ├── hooks/          # Custom Lynx hooks
│   │   ├── lib/            # Utilities and configurations
│   │   └── styles/         # CSS and styling
│   ├── convex/             # Convex backend functions
│   │   ├── schema.ts       # Database schema
│   │   ├── ragas.ts        # Raga operations
│   │   ├── musicGeneration.ts # AI music generation
│   │   ├── files.ts        # File operations
│   │   └── ml_integration.ts # ML API integration
│   └── package.json        # Frontend dependencies
├── ml/                     # Machine learning components
│   ├── cloud_run_app.py    # FastAPI ML API
│   ├── raga_classifier.py  # Raga classification model
│   └── data_loader.py      # Data loading utilities
├── scripts/                # Utility scripts
│   ├── download_training_data.py # Download ML data
│   └── upload_models_to_gcs.py  # Upload models to GCS
├── docs/                   # Documentation (this folder)
├── tests/                  # Test files
└── README.md              # Main project README
```

## **Key Benefits**

### **Modern Architecture**
- **Cross-platform development** with Lynx
- **Real-time capabilities** out of the box
- **Type-safe development** with TypeScript
- **Scalable ML hosting** with Google Cloud Run

### **Cost Effective**
- **Free tier** covers most usage
- **Pay-per-use** pricing model
- **No idle costs** with serverless architecture
- **Automatic scaling** based on demand

### **Developer Experience**
- **Hot reload** for instant development
- **Built-in authentication** and file storage
- **Comprehensive documentation**
- **Easy deployment** with one-command scripts

### **Production Ready**
- **Auto-scaling** and load balancing
- **Built-in monitoring** and logging
- **Error handling** and recovery
- **Security** best practices

## **Useful Links**

### **Development**
- **Frontend**: http://localhost:3000 (development)
- **Convex Dashboard**: https://dashboard.convex.dev/
- **Google Cloud Console**: https://console.cloud.google.com/

### **Documentation**
- **Convex Docs**: https://docs.convex.dev/
- **Google Cloud Run**: https://cloud.google.com/run/docs
- **Lynx Docs**: https://docs.lynx.dev/
- **Tailwind CSS**: https://tailwindcss.com/

## **Contributing**

1. **Fork** the repository
2. **Create** a feature branch
3. **Make** your changes
4. **Test** thoroughly
5. **Submit** a pull request

## **License**

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

## **Support**

- **Issues**: Create an issue on GitHub
- **Discussions**: Use GitHub Discussions
- **Documentation**: Check this docs folder

---

**Ready to create beautiful Indian classical music with AI!**
