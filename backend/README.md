# Backend API (FastAPI)

This directory contains the FastAPI backend for the Raga Detector project. It provides REST endpoints for raga detection, analysis, search, and CRUD operations on raga-related entities.

## Features
- Raga detection from audio (integrates with ML module in ../ml/)
- Raga analysis and comparison
- CRUD for artists, performances, audio samples, ragas, etc.
- Database integration (SQLAlchemy, async)

## How to Run

1. **Install dependencies:**
   ```sh
   pip install -r ../requirements.txt
   pip install -r ../requirements_ml.txt  # For ML endpoints
   ```
2. **Start the server:**
   ```sh
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```
   (from the backend/ directory)
3. **API docs:**
   - Visit [http://localhost:8000/docs](http://localhost:8000/docs)

## API Endpoints
- `/api/raga-detect` — Raga detection from audio
- `/api/audio-analysis` — Audio feature extraction
- `/api/raga-compare` — Compare ragas
- `/api/raga-analysis` — Detailed raga analysis
- `/api/raga` — CRUD for ragas
- `/api/artist` — CRUD for artists
- `/api/performance` — CRUD for performances
- `/api/audio-sample` — CRUD for audio samples

## ML Integration
- The backend imports the ML model from the top-level `ml/` directory.
- See [../ml/README.md](../ml/README.md) for details on training and using the ML model.

## Database
- Uses SQLAlchemy (async) for database access.
- Models are in `backend/models/`.

## Development
- All backend code is in this directory.
- Update imports to use `backend.` and `ml.` as needed.

---

# Database Overview

## Table Row Counts (as of latest seed)

- ragas: 5,893
- artists: 1,274
- audio_samples: 74
- composers: 438
- performances: 0
- talas: 43
- types: 100
- songs: 10,672

## Entity-Relationship Diagram

```mermaid
erDiagram
    SONGS ||--o{ RAGAS : raga_id
    SONGS ||--o{ COMPOSERS : composer_id
    SONGS ||--o{ TYPES : type_id
    SONGS ||--o{ TALAS : tala_id
    SONGS ||--o{ AUDIO_SAMPLES : audio_sample_id
    PERFORMANCES ||--o{ SONGS : song_id
    PERFORMANCES ||--o{ ARTISTS : artist_id
    RAGAS ||--o{ AUDIO_SAMPLES : raga_id
    ARTISTS {
        int id
        string name
    }
    RAGAS {
        int id
        string name
    }
    SONGS {
        int id
        string title
        int raga_id
        int composer_id
        int type_id
        int tala_id
        int audio_sample_id
    }
    COMPOSERS {
        int id
        string name
    }
    TYPES {
        int id
        string name
    }
    TALAS {
        int id
        string name
    }
    AUDIO_SAMPLES {
        int id
        string file_path
        int raga_id
    }
    PERFORMANCES {
        int id
        int song_id
        int artist_id
    }
```

---

See the root [README.md](../README.md) for project structure and more details. 