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

# RagaSense - AI-Powered Indian Classical Music Raga Detection

[![Lynx](https://img.shields.io/badge/Lynx-Framework-blue.svg)](https://lynxjs.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0-blue.svg)](https://www.typescriptlang.org/)
[![Convex](https://img.shields.io/badge/Convex-Backend-green.svg)](https://convex.dev/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Python-red.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Stars](https://img.shields.io/github/stars/adhit-r/RagaSense?style=social)](https://github.com/adhit-r/RagaSense)
[![Forks](https://img.shields.io/github/forks/adhit-r/RagaSense?style=social)](https://github.com/adhit-r/RagaSense)

> **Discover and analyze Indian classical music ragas using advanced AI technology**

A comprehensive full-stack application for detecting and analyzing Indian classical music ragas using machine learning, featuring a modern Lynx frontend and Convex real-time database.

## 🎵 Features

- **Real-time Raga Detection**: Upload audio files or record live to identify ragas instantly
- **Cross-platform Frontend**: Beautiful Sazhaam-like UX built with Lynx framework (Web, iOS, Android)
- **Real-time Database**: Convex integration for live data synchronization
- **User Management**: Complete authentication and user profiles
- **Detection History**: Track and analyze your raga detection results
- **Music Generation**: AI-powered music creation (coming soon)
- **Analytics**: Comprehensive usage tracking and insights

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Lynx Frontend │    │   Convex Backend │    │  FastAPI ML     │
│   (Web/iOS/     │◄──►│   (Database +    │◄──►│  Backend        │
│    Android)     │    │    Auth + Files) │    │  (Local)        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Technology Stack

- **Frontend**: Lynx + ReactLynx + TypeScript + Tailwind CSS
- **Database**: Convex (real-time, serverless)
- **Authentication**: Convex Auth
- **ML Backend**: FastAPI + Python + TensorFlow + Librosa
- **Build Tool**: Rspeedy
- **Package Manager**: Bun

## 🚀 Quick Start

### Prerequisites

- Node.js 18 or later
- Python 3.9 or later
- Bun package manager
- Git

### 1. Clone and Setup

```bash
git clone https://github.com/adhit-r/RagaSense.git
cd raga_detector
```

### 2. Backend Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Initialize database
python init_db.py

# Start backend server
python -m backend.main
```

### 3. Frontend Setup

```bash
cd frontend

# Install dependencies
bun install

# Set up Convex
bun add -g convex
convex login
convex dev --configure
convex deploy

# Configure environment
cp env.example .env.local
# Edit .env.local with your Convex URL

# Start development server
bun run dev
```

### 4. Test the System

```bash
# Test raga detection
python scripts/test_raga_detection.py

# Or use the complete system
./run_raga_detection.sh
```

## 📁 Project Structure

```
raga_detector/
├── backend/                 # FastAPI backend
│   ├── api/                # API endpoints
│   ├── models/             # Database models
│   └── main.py             # FastAPI app
├── frontend/               # Lynx frontend
│   ├── convex/             # Convex database & functions
│   ├── src/                # ReactLynx components
│   └── rspeedy.config.ts   # Build configuration
├── ml/                     # Machine learning
│   └── working_raga_detector.py
├── docs/                   # Documentation
├── scripts/                # Utility scripts
└── tests/                  # Test files
```

## 🎯 Key Components

### Raga Detection System
- **ML Model**: RandomForest classifier with audio feature extraction
- **Features**: MFCCs, Chroma, Spectral features, ZCR, RMS energy
- **Supported Ragas**: Yaman, Bhairav, Kafi (expandable)
- **Accuracy**: Currently using synthetic data (ready for real data)

### Frontend Features
- **Cross-platform**: Single codebase for Web, iOS, Android
- **Sazhaam-like UX**: Modern, intuitive interface
- **Real-time**: Live updates with Convex
- **Authentication**: Complete user management
- **File Upload**: Drag-and-drop audio file support

### Database Schema
- **Users**: Profiles and authentication
- **Ragas**: Metadata and information
- **Detections**: History and results
- **Files**: Audio file management
- **Analytics**: Usage tracking

## 🔧 Development

### Backend Development
```bash
# Start backend
python -m backend.main

# Run tests
python scripts/test_raga_detection.py

# Check database
python check_db.py
```

### Frontend Development
```bash
cd frontend

# Development server
bun run dev

# Build for platforms
bun run build:web
bun run build:ios
bun run build:android

# Convex functions
bun run convex:dev
bun run convex:deploy
```

### Database Management
```bash
cd frontend

# Deploy schema changes
convex deploy

# View data
convex dashboard
```

## 📊 Current Status

### ✅ Completed
- [x] FastAPI backend with ML integration
- [x] Lynx frontend with Sazhaam-like UX
- [x] Convex database and authentication
- [x] Raga detection system (synthetic data)
- [x] File upload and processing
- [x] User management and settings
- [x] Detection history and analytics
- [x] Cross-platform support

### 🚧 In Progress
- [ ] Real training data integration
- [ ] Music generation features
- [ ] Advanced analytics dashboard
- [ ] Mobile app optimization

### 📋 Planned
- [ ] More raga support
- [ ] Advanced ML models
- [ ] Social features
- [ ] Performance optimization

## 🗺️ Development Roadmap

We have a comprehensive [development roadmap](docs/ROADMAP.md) with 23 detailed tasks organized into 4 phases:

### **Phase 1: Foundation & Core Features** (Q1 2024)
- ML model enhancement with real training data
- Frontend polish and mobile app development
- Complete Convex integration

### **Phase 2: Advanced Features** (Q2 2024)
- AI music generation capabilities
- Social and collaborative features
- Advanced analytics dashboard

### **Phase 3: Enterprise & Scale** (Q3 2024)
- Multi-tenant architecture
- Performance optimization
- Scalability improvements

### **Phase 4: Innovation & Research** (Q4 2024)
- Deep learning integration
- Educational platform
- Research collaboration

## 📊 Project Management

Track development progress with our [GitHub Project Board](https://github.com/adhit-r/RagaSense/projects):

- **Visual Issue Tracking** with Kanban board
- **Milestone Management** for each development phase
- **Automated Workflows** for issue lifecycle
- **Progress Analytics** and velocity tracking

[Set up the project board](docs/PROJECT_BOARD_SETUP.md) to start contributing!

## 🚀 Deployment

### Local Development
```bash
# Complete system
./run_raga_detection.sh

# Or individual components
python -m backend.main &  # Backend
cd frontend && bun run dev  # Frontend
```

### Production Deployment
```bash
# Backend (deploy to your preferred service)
# Heroku, Railway, DigitalOcean, etc.

# Frontend - Multiple deployment options available:
cd frontend
bun run build:web

# Option 1: Netlify (Recommended)
netlify deploy --prod --dir=dist/web

# Option 2: Vercel
vercel --prod

# Option 3: Firebase
firebase deploy --only hosting

# Option 4: GitHub Pages (automatic via GitHub Actions)
# Just push to main branch

# Option 5: Docker
docker build -f deploy/Dockerfile -t ragasense-frontend .
docker run -p 80:80 ragasense-frontend

# Option 6: Automated Script (Easiest)
./frontend/deploy.sh netlify    # Deploy to Netlify
./frontend/deploy.sh vercel     # Deploy to Vercel
./frontend/deploy.sh firebase   # Deploy to Firebase
./frontend/deploy.sh github     # Deploy to GitHub Pages
./frontend/deploy.sh docker     # Deploy with Docker
./frontend/deploy.sh railway    # Deploy to Railway

# Convex (already deployed)
bun run convex:deploy
```

**See [Deployment Guide](docs/DEPLOYMENT_GUIDE.md) for detailed instructions.**

## 📚 Documentation

- [API Documentation](docs/API_DOCS.md)
- [ML System Details](docs/ML_RAGA_DETECTION_SCIENTIFIC.md)
- [Codebase Organization](docs/CODEBASE_ORGANIZATION.md)
- [Working System Guide](docs/WORKING_RAGA_DETECTION_SYSTEM.md)
- [Frontend Setup](frontend/README.md)
- [Development Roadmap](docs/ROADMAP.md)
- [Project Board Setup](docs/PROJECT_BOARD_SETUP.md)
- [Deployment Guide](docs/DEPLOYMENT_GUIDE.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

See [Contributing Guide](.github/CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Indian classical music community
- Open source contributors
- Lynx framework team
- Convex team

## 🔗 Links

- **Repository**: https://github.com/adhit-r/RagaSense
- **Issues**: https://github.com/adhit-r/RagaSense/issues
- **Discussions**: https://github.com/adhit-r/RagaSense/discussions
- **Wiki**: https://github.com/adhit-r/RagaSense/wiki

---

**Built with ❤️ for Indian classical music enthusiasts** 🎵✨