# FER Benchmarking Platform

**Facial Emotion Recognition (FER) Benchmarking Platform** is a tool designed to evaluate and compare the performance of different emotion recognition models. It provides a user-friendly interface to run predictions on images using both custom-trained models and state-of-the-art pretrained models, as well as visualize their performance statistics.

## Key Features

### 1. Image Processing & Prediction
Run emotion recognition models on your own images:
- **Dual Model Inference**: Simultaneously run predictions using:
    - **Custom Model**: A CNN trained on the FER-2013 dataset.
    - **Pretrained Model**: A Hugging Face pipeline model (e.g., `michellejieli/emotion_text_classifier` adapted for images or similar).
- **Interactive UI**: Upload an image to see a preview and get immediate results.
- **Detailed Analysis**: View the top predicted emotion, confidence score, and a visual probability distribution for 7 emotions (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral).

### 2. Model Statistics & Comparison
Analyze and compare model performance (accessible via API/Future UI):
- **Performance Metrics**: View confusion matrices, ROC curves, and per-class precision/recall charts.
- **Before/After Analysis**: Compare the accuracy of the custom model before and after training.
- **aggregated Stats**: Endpoint access to comprehensive evaluation reports.

---

## Installation & Setup

### Prerequisites
- **Python 3.8+**
- **Node.js & npm** (for the frontend)
- **Anaconda** (optional, for environment management)

### 1. Backend Setup (FastAPI)

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd FERBenchmarkingPlatform
    ```

2.  **Create and activate the environment**:
    Using Conda (recommended):
    ```bash
    conda env create -f env.yml
    conda activate <model-env-name>
    ```
    *Alternatively, install dependencies manually:*
    ```bash
    pip install -r CustomModel/requirements.txt
    ```

3.  **Run the Backend Server**:
    start the FastAPI server from the root directory:
    ```bash
    python server.py
    ```
    The API will be available at `http://localhost:8000`.

### 2. Frontend Setup (Angular)

1.  **Navigate to the frontend directory**:
    ```bash
    cd frontend
    ```

2.  **Install dependencies**:
    ```bash
    npm install
    ```

3.  **Run the Frontend**:
    ```bash
    npm start
    ```
    The application will be accessible at `http://localhost:4200`.

---

## Usage Guide

1.  **Start both the Backend and Frontend** servers using the commands above.
2.  Open your browser to **`http://localhost:4200`**.
3.  **Image Processing Page**:
    - Click **"Upload Image"** to select a face image.
    - Click **"Run Models"** to process the image.
    - Compare the results (Top Emotion, Confidence, Probability Bars) between the Pretrained and Custom models side-by-side.
4.  **Comparison Page**:
    - Use the navigation bar to switch to the "Comparison" view (Feature coming soon).

## Project Structure

- `server.py`: Main entry point for the FastAPI backend.
- `frontend/`: Angular source code for the user interface.
- `custom_model/`: Contains `train.py`, `inference.py`, and model artifacts.
- `pretrained_model/`: Contains the interface for the Hugging Face model.
- `evaluation_scripts/`: Scripts for evaluating models and generating plots/reports.
- `stats_api.py`: API endpoints for serving investigation statistics.
