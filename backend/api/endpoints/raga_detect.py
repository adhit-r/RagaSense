from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import Optional
import os
import tempfile
import uuid
from pathlib import Path
from datetime import datetime

from ...services.raga_detector import raga_detection_service

router = APIRouter(prefix="/ragas", tags=["raga_detection"])

# Create upload directory if it doesn't exist
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

def save_upload_file(upload_file: UploadFile, destination: Path) -> str:
    """Save uploaded file to disk and return the file path."""
    try:
        # Generate unique filename
        file_ext = Path(upload_file.filename).suffix.lower()
        unique_id = str(uuid.uuid4())
        file_path = destination / f"{unique_id}{file_ext}"
        
        # Save file
        with file_path.open("wb") as buffer:
            buffer.write(upload_file.file.read())
            
        return str(file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

@router.post("/detect", summary="Detect raga from audio")
async def detect_raga(
    audio: UploadFile = File(..., description="Audio file to analyze (WAV, MP3, etc.)"),
    duration: Optional[int] = 30,
):
    """
    Detect the raga from an audio file.
    
    - **audio**: Audio file to analyze
    - **duration**: Maximum duration in seconds to process (default: 30)
    """
    try:
        # Validate file type
        allowed_extensions = {'.wav', '.mp3', '.ogg', '.flac', '.m4a'}
        file_ext = Path(audio.filename).suffix.lower()
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Allowed types: {', '.join(allowed_extensions)}"
            )
        
        # Check if service is ready
        if not raga_detection_service.is_ready():
            raise HTTPException(
                status_code=503,
                detail="Raga detection service is not ready. Please try again later."
            )
        
        # Save the uploaded file temporarily
        temp_file = None
        try:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
                # Save the uploaded file
                contents = await audio.read()
                temp_file.write(contents)
                temp_file_path = temp_file.name
            
            # Get prediction
            result = raga_detection_service.predict(temp_file_path)
            
            # Add request metadata
            result['request_info'] = {
                'filename': audio.filename,
                'file_size': len(contents),
                'file_type': file_ext,
                'processed_at': datetime.utcnow().isoformat(),
                'duration_limit': duration
            }
            
            return result
            
        finally:
            # Clean up the temporary file
            if temp_file and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")

@router.get("/supported-ragas", summary="Get list of supported ragas")
async def get_supported_ragas():
    """
    Get a list of all ragas that the model can detect.
    """
    try:
        ragas = raga_detection_service.get_supported_ragas()
        
        return {
            "success": True,
            "data": {
                "total_ragas": len(ragas),
                "ragas": ragas,
                "model_info": raga_detection_service.get_model_info()
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving supported ragas: {str(e)}")

@router.get("/model-info", summary="Get model information")
async def get_model_info():
    """
    Get information about the current raga detection model.
    """
    try:
        model_info = raga_detection_service.get_model_info()
        
        return {
            "success": True,
            "data": model_info,
            "service_status": {
                "ready": raga_detection_service.is_ready(),
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving model info: {str(e)}")

@router.get("/health", summary="Health check")
async def health_check():
    """
    Check the health status of the raga detection service.
    """
    try:
        is_ready = raga_detection_service.is_ready()
        model_info = raga_detection_service.get_model_info()
        
        return {
            "success": True,
            "status": "healthy" if is_ready else "degraded",
            "service_ready": is_ready,
            "model_info": model_info,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {
            "success": False,
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }
