from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api.endpoints import raga, artist, performance, audio_sample
from backend.api.endpoints import raga_detect_router, audio_analysis_router, raga_compare_router, raga_analysis_router

app = FastAPI(title="RagaSense API", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(raga.router)
app.include_router(artist.router)
app.include_router(performance.router)
app.include_router(audio_sample.router)
app.include_router(raga_detect_router)
app.include_router(audio_analysis_router)
app.include_router(raga_compare_router)
app.include_router(raga_analysis_router) 