import sys
import os
sys.path.append("/Users/adhi/Documents/learn/raga_detector")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.db import AsyncSessionLocal
from app.models.raga import Raga
from sqlalchemy.future import select
from typing import List
import shutil
from backend.models.raga_classifier import RagaClassifier
from fastapi.responses import JSONResponse
from fastapi import status

router = APIRouter()

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), '../../../uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@router.get('/ragas', response_model=List[str])
async def list_ragas(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Raga.name))
    ragas = [row[0] for row in result.all()]
    return ragas

@router.post('/predict')
async def predict_raga(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail='No file uploaded')
    filename = file.filename
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    with open(filepath, 'wb') as buffer:
        shutil.copyfileobj(file.file, buffer)
    try:
        classifier = RagaClassifier()
        result = classifier.predict_raga(filepath)
        os.remove(filepath)
        if 'error' in result:
            return JSONResponse(status_code=500, content={
                'success': False,
                'error': 'Prediction failed',
                'message': result['error']
            })
        return {
            'success': True,
            'predicted_raga': result.get('raga'),
            'confidence': result.get('confidence', {}),
            'raga_info': result.get('raga_info', {})
        }
    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        raise HTTPException(status_code=500, detail=str(e)) 