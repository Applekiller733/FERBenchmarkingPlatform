from fastapi import FastAPI, UploadFile, File, HTTPException, APIRouter
from transformers import pipeline
from PIL import Image
import io
import asyncio

router = APIRouter()

# Initialize the pipeline
classifier = pipeline("image-classification", model="dima806/facial_emotions_image_detection")


@router.post("/predict_pretrained", tags=["Pretrained Model"])
async def predict_emotions(images: list[UploadFile] = File(...)):
    """
    Endpoint to receive a list of images and return predictions for each using the Pretrained ViT model.
    """
    if not images:
        raise HTTPException(status_code=400, detail="No images provided")

    image_objects = []
    filenames = []

    for img in images:
        try:
            content = await img.read()
            image = Image.open(io.BytesIO(content)).convert("RGB")
            image_objects.append(image)
            filenames.append(img.filename)
        except Exception as e:
            # If one image fails, we keep the index alignment
            image_objects.append(None)
            filenames.append(img.filename)

    # 2. Run Batch Prediction
    valid_images = [img for img in image_objects if img is not None]

    if not valid_images:
        raise HTTPException(status_code=400, detail="No valid images were uploaded")

    all_predictions = classifier(valid_images)

    # 3. Format results
    final_results = []
    pred_idx = 0

    for i, filename in enumerate(filenames):
        if image_objects[i] is None:
            final_results.append({"filename": filename, "error": "Invalid image file"})
            continue

        preds = all_predictions[pred_idx]
        top_prediction = preds[0]

        final_results.append({
            "filename": filename,
            "top_emotion": top_prediction['label'],
            "confidence": round(top_prediction['score'], 4),
            "full_analysis": {p['label']: round(p['score'], 4) for p in preds}
        })
        pred_idx += 1

    return {
        "count": len(final_results),
        "results": final_results
    }


# Create the app only if running standalone or for testing imports that expect 'app'
app = FastAPI(title="ViT Emotion Recognition API")
app.include_router(router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)