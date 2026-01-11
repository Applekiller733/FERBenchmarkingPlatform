import sys
import os
from transformers import pipeline
from torchvision import datasets
from tqdm import tqdm
from PIL import Image
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import plot_confusion_matrix, save_classification_report, plot_per_class_metrics, plot_multiclass_roc

def evaluate_pretrained_model():
    # Configuration
    TEST_DATA_DIR = os.path.join("custom_model", "data", "test")
    OUTPUT_DIR = os.path.join("evaluation_scripts", "results")
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print("Loading pretrained pipeline...")
    # top_k=None ensures we get scores for all classes
    classifier = pipeline("image-classification", model="dima806/facial_emotions_image_detection")

    if not os.path.exists(TEST_DATA_DIR):
        print(f"Error: Test data directory not found at {TEST_DATA_DIR}")
        return

    test_dataset = datasets.ImageFolder(TEST_DATA_DIR)
    classes = test_dataset.classes
    print(f"Dataset Classes: {classes}")

    all_preds_labels = []
    all_true_labels = []
    all_probs = []

    print("Starting evaluation...")
    for path, target_idx in tqdm(test_dataset.samples, desc="Evaluating"):
        try:
            predictions = classifier(path, top_k=None)
            
            top_prediction_label = predictions[0]['label']
            if top_prediction_label in classes:
                all_preds_labels.append(classes.index(top_prediction_label))
            else:
                 try:
                     all_preds_labels.append(classes.index(top_prediction_label))
                 except ValueError:
                     # If unknown label, we might have issues.
                     print(f"Warning: Unknown label {top_prediction_label}")
                     all_preds_labels.append(-1)
            
            probs_vec = [0.0] * len(classes)
            for pred in predictions:
                label = pred['label']
                score = pred['score']
                if label in classes:
                    idx = classes.index(label)
                    probs_vec[idx] = score
            
            all_probs.append(probs_vec)
            all_true_labels.append(target_idx)
                
        except Exception as e:
            print(f"Error processing {path}: {e}")

    # Check for mismatches
    unique_preds = set(all_preds_labels)
    unique_true = set(all_true_labels)
    combined_classes = sorted(list(unique_preds.union(unique_true)))
    
    print(f"Unique Predicted Labels: {unique_preds}")
    print(f"Unique True Labels: {unique_true}")
    
    # Check if we have probabilities for all samples
    if len(all_probs) != len(all_true_labels):
        print("Warning: Probability collection mismatch.")

    # Convert to numpy for ROC
    y_probs_np = np.array(all_probs)

    plot_confusion_matrix(
        all_true_labels, 
        all_preds_labels, 
        classes, 
        os.path.join(OUTPUT_DIR, "pretrained_model_confusion_matrix.png"),
        title="Pretrained Model Confusion Matrix"
    )
    
    plot_confusion_matrix(
        all_true_labels, 
        all_preds_labels, 
        classes, 
        os.path.join(OUTPUT_DIR, "pretrained_model_confusion_matrix_normalized.png"),
        title="Pretrained Model Normalized Confusion Matrix",
        normalize=True
    )

    # Classification Report
    report_dict = save_classification_report(
        all_true_labels, 
        all_preds_labels, 
        classes, 
        os.path.join(OUTPUT_DIR, "pretrained_model_report.txt")
    )
    
    # Per-Class Metrics
    plot_per_class_metrics(
        report_dict,
        classes,
        os.path.join(OUTPUT_DIR, "pretrained_model_per_class_metrics.png"),
        title="Pretrained Model: Per-Class Performance"
    )
    
    plot_multiclass_roc(
        all_true_labels,
        y_probs_np,
        classes, # Use dataset classes as reference for probability vector
        os.path.join(OUTPUT_DIR, "pretrained_model_roc_curve.png"),
        title="Pretrained Model: One-vs-Rest ROC Curves"
    )

    print("Evaluation complete.")

if __name__ == "__main__":
    evaluate_pretrained_model()
