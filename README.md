---
title: "RagaSense - AI-Powered Indian Classical Music Raga Detection"
description: "Discover and analyze Indian classical music ragas using advanced AI technology. Upload audio files for instant raga identification with machine learning."
keywords: "raga detection, indian classical music, AI music analysis, machine learning, audio processing, carnatic music, hindustani music, music technology"
author: "RagaSense Team"
og:title: "RagaSense - AI-Powered Indian Classical Music Raga Detection"
og:description: "Discover and analyze Indian classical music ragas using advanced AI technology. Upload audio files for instant raga identification."
og:type: "website"
og:url: "https://github.com/adhit-r/RagaSense"
twitter:card: "summary_large_image"
twitter:title: "RagaSense - AI-Powered Indian Classical Music Raga Detection"
twitter:description: "Discover and analyze Indian classical music ragas using advanced AI technology."
---

# RagaSense

AI-powered Carnatic raga detection and music generation platform with cross-platform mobile support.

## 🚀 **Features**

### **Raga Detection**
- **Live Recording**: Real-time audio recording and analysis
- **File Upload**: Support for MP3, WAV, M4A, AAC files
- **Multiple Models**: Choose from 4 detection methods
  - Local Custom Model (trained on our data)
  - Hugging Face Cloud API
  - Local Hugging Face Model (downloaded)
  - Ensemble (all models combined)
- **Visual Feedback**: Processing indicators and detailed results
- **Confidence Scores**: Detailed prediction confidence

### **Music Generation**
- **Raga Selection**: Choose from popular Carnatic ragas
- **Style Options**: Carnatic, Hindustani, Fusion
- **Duration Control**: 10-120 seconds of generated music
- **Audio Playback**: Built-in audio player
- **Download Support**: Save generated music locally

### **Cross-Platform Mobile App**
- **Web**: Progressive Web App (PWA)
- **iOS**: Native iOS app
- **Android**: Native Android app
- **Responsive Design**: Works on all screen sizes

## 🏗️ **Project Structure**

```
ragasense/
├── frontend/                    # Flutter mobile app (NEW)
│   ├── lib/                    # Dart source code
│   ├── android/                # Android-specific code
│   ├── ios/                    # iOS-specific code
│   ├── web/                    # Web-specific code
│   └── pubspec.yaml           # Flutter dependencies
├── backend/                    # FastAPI backend
│   ├── main.py                # Main FastAPI server
│   ├── models/                # ML model implementations
│   └── tests/                 # Backend tests
├── ml/                        # Machine Learning pipeline
│   ├── data/                  # Data processing
│   ├── models/                # Model training
│   ├── evaluation/            # Model evaluation
│   └── README.md              # ML documentation
├── docs/                      # Documentation
├── scripts/                   # Utility scripts
├── external_data/             # External datasets
├── archive/                   # Archived components
│   └── frontend/              # Old React frontend (ARCHIVED)
└── start.sh                   # Quick start script
```

## 🛠️ **Technology Stack**

### **Frontend (Flutter)**
- **Flutter 3.16+** - Cross-platform UI framework
- **Dart 3.2+** - Programming language
- **Material 3** - Modern design system
- **Riverpod** - State management
- **Audio packages** - Recording, playback, file picking

### **Backend (Python)**
- **FastAPI** - Modern web framework
- **PyTorch** - Deep learning framework
- **librosa** - Audio processing
- **SQLAlchemy** - Database ORM
- **Pydantic** - Data validation

### **Machine Learning**
- **ResNet** - Deep learning architecture
- **Hugging Face** - Pre-trained models
- **Ensemble Methods** - Multiple model combination
- **Audio Feature Extraction** - MFCC, Chroma, Spectral features

### **Database & Storage**
- **Convex** - Real-time database
- **SQLite** - Local storage
- **File System** - Audio file storage

## 🚀 **Quick Start**

### **Prerequisites**
- Python 3.10+
- Flutter SDK 3.16+
- Bun (for package management)
- Git

### **1. Clone Repository**
```bash
git clone <repository-url>
cd ragasense
```

### **2. Start Backend**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

### **3. Start Frontend**
```bash
cd frontend
flutter pub get
flutter run -d chrome  # For web
flutter run -d ios      # For iOS
flutter run -d android  # For Android
```

### **4. Quick Start Script**
```bash
./start.sh  # Starts both backend and frontend
```

## 📱 **Platform Support**

| Platform | Status | Notes |
|----------|--------|-------|
| Web | ✅ Supported | PWA with audio recording |
| iOS | ✅ Supported | Native audio recording |
| Android | ✅ Supported | Native audio recording |
| macOS | 🔄 Planned | Desktop support |
| Windows | 🔄 Planned | Desktop support |
| Linux | 🔄 Planned | Desktop support |

## 🎯 **ML Architecture**

### **Model Options**
1. **Local Custom Model**: Trained on our Convex dataset
2. **Hugging Face Cloud**: `jeevster/carnatic-raga-classifier`
3. **Local Hugging Face**: Downloaded model for offline use
4. **Ensemble**: Combines all models for best accuracy

### **Data Sources**
- **CompMusic**: High-quality Carnatic music dataset
- **Saraga**: Indian classical music collections
- **Sanidha**: Georgia Tech's raga dataset
- **Convex Database**: Our curated raga metadata
- **External Repositories**: Community datasets

### **Feature Engineering**
- **MFCC**: Mel-frequency cepstral coefficients
- **Chroma**: Pitch class profiles
- **Spectral**: Spectral centroid, rolloff, bandwidth
- **Rhythm**: Tempo, beat tracking
- **Harmonic**: Tonnetz, harmonic features

## 🔧 **Configuration**

### **Environment Variables**
```env
# Backend
BACKEND_PORT=8002
MODEL_PATH=./ml/models/
DATA_PATH=./external_data/

# Frontend
BACKEND_URL=http://localhost:8002
```

### **Model Configuration**
```python
# backend/main.py
MODEL_CONFIG = {
    "num_classes": 528,
    "sample_rate": 44100,
    "clip_length": 30,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}
```

## 🧪 **Testing**

### **Backend Tests**
```bash
cd backend
python -m pytest tests/
```

### **Frontend Tests**
```bash
cd frontend
flutter test
```

### **End-to-End Tests**
```bash
cd scripts
python test_system.py
```

## 📊 **Performance**

- **Detection Accuracy**: 85%+ on test dataset
- **Processing Time**: 2-5 seconds per audio clip
- **Model Size**: ~50MB (compressed)
- **Memory Usage**: <500MB RAM
- **Cross-Platform**: Single codebase for all platforms

## 🚀 **Deployment**

### **Backend Deployment**
```bash
# Using uvicorn
uvicorn main:app --host 0.0.0.0 --port 8002

# Using Docker
docker build -t ragasense-backend .
docker run -p 8002:8002 ragasense-backend
```

### **Frontend Deployment**
```bash
# Web
cd frontend
flutter build web
# Deploy build/web folder

# Mobile
flutter build appbundle --release  # Android
flutter build ios --release        # iOS
```

## 📈 **Roadmap**

### **Phase 1: Core Features** ✅
- [x] Basic raga detection
- [x] Multiple model support
- [x] Cross-platform mobile app
- [x] Audio recording and upload

### **Phase 2: Enhanced Features** 🔄
- [ ] Music generation
- [ ] User authentication
- [ ] Learning mode
- [ ] Practice sessions

### **Phase 3: Advanced Features** 📋
- [ ] Real-time detection
- [ ] Advanced explainability
- [ ] Social features
- [ ] Mobile app stores

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch
3. Follow coding standards
4. Add tests for new features
5. Submit a pull request

## 📄 **License**

MIT License - see [LICENSE](LICENSE) for details.

## 🎵 **About Carnatic Music**

Carnatic music is a classical music tradition from South India, characterized by its complex melodic structures called "ragas" and rhythmic patterns called "talas". This application helps musicians and enthusiasts identify and understand these intricate musical patterns using AI.

---

**Built with ❤️ for the Carnatic music community**