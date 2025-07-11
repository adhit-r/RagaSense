from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="RagaSense API", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers will be included here as they are implemented
from app.api.endpoints import router as raga_router
app.include_router(raga_router, prefix="/api", tags=["Ragas"])
# app.include_router(artist.router, prefix="/api/artists", tags=["Artists"])
# app.include_router(performance.router, prefix="/api/performances", tags=["Performances"])
# app.include_router(audio_sample.router, prefix="/api/audio-samples", tags=["Audio Samples"])
# app.include_router(analysis.router, prefix="/api/analysis", tags=["Analysis"])
