# Backend API

Clean FastAPI backend for RagaSense.

## Structure

```
backend/
├── api/                    # API routes
│   ├── raga_detection.py  # Raga detection endpoints
│   └── music_generation.py # Music generation endpoints
├── core/                   # Core backend logic
│   ├── config.py          # Configuration
│   └── database.py        # Database connections
├── schemas/                # Pydantic models
│   └── models.py          # Request/Response models
└── main.py                 # Main FastAPI application
```

## Quick Start

1. **Install dependencies**: `pip install fastapi uvicorn`
2. **Start server**: `python main.py`
3. **API docs**: http://localhost:8002/docs

## Endpoints

- `POST /api/detect-raga` - Detect raga from audio
- `POST /api/generate-music` - Generate music in a raga
- `GET /health` - Health check

## ML Integration

The backend connects to ML services in the `ml/` directory for:
- Raga detection using multiple models
- Audio processing and feature extraction
- Model inference and predictions 