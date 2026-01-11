import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize

def plot_confusion_matrix(y_true, y_pred, classes, output_path, title="Confusion Matrix", normalize=False):
    """
    Plots the confusion matrix.
    normalize: If True, plots the normalized confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved confusion matrix to {output_path}")

def save_classification_report(y_true, y_pred, classes, output_path):
    """
    Generates and saves the classification report.
    Returns the dictionary format for further plotting.
    """
    report_dict = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    report_str = classification_report(y_true, y_pred, target_names=classes)
    
    with open(output_path, "w") as f:
        f.write(report_str)
    print(f"Saved classification report to {output_path}")
    return report_dict

def plot_per_class_metrics(report_dict, classes, output_path, title="Per-Class Performance Metrics"):
    """
    Plots Precision, Recall, and F1-Score for each class.
    """
    metrics = ['precision', 'recall', 'f1-score']
    data = []
    
    for cls in classes:
        if cls in report_dict:
            for metric in metrics:
                data.append({
                    'Class': cls,
                    'Metric': metric.capitalize(),
                    'Score': report_dict[cls][metric]
                })
    
    import pandas as pd
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='Class', y='Score', hue='Metric', palette='viridis')
    plt.title(title)
    plt.ylim(0, 1.1)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved per-class metrics plot to {output_path}")

def plot_multiclass_roc(y_true, y_score, classes, output_path, title="Multiclass ROC Curves"):
    """
    Plots One-vs-Rest ROC curves for each class.
    y_true: Ground truth labels (indices or names converted to indices)
    y_score: Predicted probabilities (n_samples, n_classes)
    """
    # Binarize the output
    n_classes = len(classes)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
    plt.figure(figsize=(10, 8))
    
    colors = plt.cm.get_cmap('tab10', n_classes)
    
    for i, color in zip(range(n_classes), colors.colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'{classes[i]} (AUC = {roc_auc[i]:.2f})')
                 
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved ROC curves to {output_path}")

def plot_model_comparison(accuracies, labels, output_path, title="Model Accuracy Comparison"):
    """
    Plots a simple bar chart comparing accuracies.
    accuracies: list of float values
    labels: list of string labels for the bars
    """
    plt.figure(figsize=(8, 6))
    sns.barplot(x=labels, y=accuracies, palette='magma')
    plt.title(title)
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.0)
    
    # Add text labels on bars
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.01, f"{v:.2%}", ha='center', va='bottom')
        
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved comparison plot to {output_path}")
