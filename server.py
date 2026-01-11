from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from custom_model.api import router as custom_router
from pretrained_model.app import router as pretrained_router
from stats_api import router as stats_router
import os

app = FastAPI(title="FER Benchmarking Platform")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Mount custom model router
app.include_router(custom_router, prefix="/custom", tags=["Custom Model"])

# Mount pretrained model router
app.include_router(pretrained_router, prefix="/pretrained", tags=["Pretrained Model"])

# Mount stats router
app.include_router(stats_router, prefix="/stats", tags=["Stats"])

# Mount Static Files for Results
results_dir = os.path.join("evaluation_scripts", "results")
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
    
app.mount("/static/results", StaticFiles(directory=results_dir), name="results")

@app.get("/")
def read_root():
    return {"message": "Welcome to FER Benchmarking Platform"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
