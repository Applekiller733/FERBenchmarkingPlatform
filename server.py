from fastapi import FastAPI
import uvicorn
from custom_model.api import router as custom_router
from pretrained_model.app import router as pretrained_router

app = FastAPI(title="FER Benchmarking Platform")

# Mount custom model router
app.include_router(custom_router, prefix="/custom", tags=["Custom Model"])

# Mount pretrained model router
app.include_router(pretrained_router, prefix="/pretrained", tags=["Pretrained Model"])

@app.get("/")
def read_root():
    return {"message": "Welcome to FER Benchmarking Platform"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
