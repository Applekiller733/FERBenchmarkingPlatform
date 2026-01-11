from fastapi import APIRouter, UploadFile, File, HTTPException
from .inference import EmotionPredictor

router = APIRouter()
predictor = EmotionPredictor()

@router.post("/predict_custom")
async def predict_custom(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    content = await file.read()
    result = predictor.predict(content)
    
    if result is None:
        raise HTTPException(status_code=500, detail="Prediction failed")
        
    return {
        "filename": file.filename,
        "result": result
    }
