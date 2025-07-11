# Raga Detector: Project Overview

This repository contains a full-stack application for Indian classical raga detection, analysis, and search. The system is composed of three main components:

- **ML Module** (`ml/`): Deep learning model for raga classification from audio.
- **Backend API** (`backend/`): FastAPI backend serving raga data, detection, and analysis endpoints.
- **Frontend** (`frontend/`): React web app for user interaction and audio upload.

---

## Project Structure

```
raga_detector/
├── backend/         # FastAPI backend (API, DB, models, etc.)
├── ml/              # ML model, training, and test scripts
├── frontend/        # React frontend
├── requirements.txt
├── requirements_ml.txt
├── run_local.sh
├── README.md
└── ... (other root files)
```

---

## How to Run Each Part

### Backend (FastAPI)
See [backend/README.md](backend/README.md)

### ML Module
See [ml/README.md](ml/README.md)

### Frontend (React)
See [frontend/README.md](frontend/README.md)

---

## About
This project enables raga detection, analysis, and search for Indian classical music using modern ML and web technologies. Each part is modular and can be run or developed independently.