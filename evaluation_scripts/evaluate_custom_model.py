import sys
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np

# Add root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from custom_model.model import FERModel
from utils import plot_confusion_matrix, save_classification_report, plot_per_class_metrics, plot_multiclass_roc, plot_model_comparison

def get_predictions(model, loader, device):
    """
    Runs inference on the loader and returns labels, predictions, and probabilities.
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

def evaluate_custom_model():
    # Configuration
    MODEL_PATH = os.path.join("custom_model", "model.pth")
    TEST_DATA_DIR = os.path.join("custom_model", "data", "test")
    OUTPUT_DIR = os.path.join("evaluation_scripts", "results")
    BATCH_SIZE = 64
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data Transforms
    test_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    if not os.path.exists(TEST_DATA_DIR):
        print(f"Error: Test data directory not found at {TEST_DATA_DIR}")
        return

    test_dataset = datasets.ImageFolder(TEST_DATA_DIR, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    classes = test_dataset.classes
    print(f"Classes found: {classes}")

    # --- 1. Evaluate Untrained Model ("Before Training") ---
    print("\n--- Evaluating Untrained Model (Before Training) ---")
    untrained_model = FERModel(num_classes=7).to(device)
    # No weights loaded, so it's initialized with random weights
    
    y_true_before, y_pred_before, _ = get_predictions(untrained_model, test_loader, device)
    
    # Calculate simple accuracy
    acc_before = np.mean(y_true_before == y_pred_before)
    print(f"Untrained Model Accuracy: {acc_before:.4f}")


    # --- 2. Evaluate Trained Model ("After Training") ---
    print("\n--- Evaluating Trained Model (After Training) ---")
    trained_model = FERModel(num_classes=7).to(device)
    
    if os.path.exists(MODEL_PATH):
        trained_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"Loaded model weights from {MODEL_PATH}")
    else:
        print(f"Error: Model file not found at {MODEL_PATH}. Cannot proceed with After metrics.")
        return

    y_true_after, y_pred_after, y_probs_after = get_predictions(trained_model, test_loader, device)
    
    acc_after = np.mean(y_true_after == y_pred_after)
    print(f"Trained Model Accuracy: {acc_after:.4f}")

    # --- 3. Generate Comparison Plot ---
    plot_model_comparison(
        [acc_before, acc_after],
        ["Before Training", "After Training"],
        os.path.join(OUTPUT_DIR, "custom_model_before_after_accuracy.png"),
        title="Custom Model Accuracy: Before vs After Training"
    )

    # --- 4. Generate Detailed Plots for Trained Model ---
    
    # Confusion Matrix
    plot_confusion_matrix(
        y_true_after, 
        y_pred_after, 
        classes, 
        os.path.join(OUTPUT_DIR, "custom_model_confusion_matrix.png"),
        title="Custom Model Confusion Matrix"
    )
    
    plot_confusion_matrix(
        y_true_after, 
        y_pred_after, 
        classes, 
        os.path.join(OUTPUT_DIR, "custom_model_confusion_matrix_normalized.png"),
        title="Custom Model Normalized Confusion Matrix",
        normalize=True
    )

    # Classification Report
    report_dict = save_classification_report(
        y_true_after, 
        y_pred_after, 
        classes, 
        os.path.join(OUTPUT_DIR, "custom_model_report.txt")
    )
    
    # Per-Class Metrics
    plot_per_class_metrics(
        report_dict,
        classes,
        os.path.join(OUTPUT_DIR, "custom_model_per_class_metrics.png"),
        title="Custom Model: Per-Class Performance"
    )
    
    # ROC Curves
    plot_multiclass_roc(
        y_true_after,
        y_probs_after,
        classes,
        os.path.join(OUTPUT_DIR, "custom_model_roc_curve.png"),
        title="Custom Model: One-vs-Rest ROC Curves"
    )

    print("Evaluation complete.")

if __name__ == "__main__":
    evaluate_custom_model()
