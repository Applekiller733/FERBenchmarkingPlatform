from fastapi import APIRouter, HTTPException
import os

router = APIRouter()

RESULTS_DIR = os.path.join("evaluation_scripts", "results")

def read_report(filename):
    path = os.path.join(RESULTS_DIR, filename)
    if os.path.exists(path):
        with open(path, "r") as f:
            return f.read()
    return None

@router.get("/custom", tags=["Stats"])
async def get_custom_stats():
    """
    Returns aggregated statistics and plot URLs for the Custom Model.
    """
    report_content = read_report("custom_model_report.txt")
    
    # Base URL for static files is hosted at /static/results/
    # We construct reliable relative paths or absolute URLs if domain is known (but relative is safer for frontend)
    base_url = "/static/results"
    
    return {
        "model": "Custom Model",
        "report_content": report_content,
        "plots": {
            "before_after_accuracy": f"{base_url}/custom_model_before_after_accuracy.png",
            "confusion_matrix": f"{base_url}/custom_model_confusion_matrix.png",
            "confusion_matrix_normalized": f"{base_url}/custom_model_confusion_matrix_normalized.png",
            "per_class_metrics": f"{base_url}/custom_model_per_class_metrics.png",
            "roc_curve": f"{base_url}/custom_model_roc_curve.png"
        }
    }

@router.get("/pretrained", tags=["Stats"])
async def get_pretrained_stats():
    """
    Returns aggregated statistics and plot URLs for the Pretrained Model.
    """
    report_content = read_report("pretrained_model_report.txt")
    
    base_url = "/static/results"
    
    return {
        "model": "Pretrained Model",
        "report_content": report_content,
        "plots": {
            "confusion_matrix": f"{base_url}/pretrained_model_confusion_matrix.png",
            "confusion_matrix_normalized": f"{base_url}/pretrained_model_confusion_matrix_normalized.png",
            "per_class_metrics": f"{base_url}/pretrained_model_per_class_metrics.png",
            "roc_curve": f"{base_url}/pretrained_model_roc_curve.png"
        }
    }

@router.get("/comparison", tags=["Stats"])
async def get_comparison_stats():
    """
    Returns simple comparison data if needed separately.
    Currently best served via the plots in /custom.
    This endpoint verifies the existence of the comparison plot.
    """
    path = os.path.join(RESULTS_DIR, "custom_model_before_after_accuracy.png")
    if os.path.exists(path):
         return {
             "comparison_plot": "/static/results/custom_model_before_after_accuracy.png",
             "message": "Comparison plot available."
         }
    return {"message": "Comparison plot not found."}
