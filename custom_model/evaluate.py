import torch
import torch.nn as nn
import os
from .model import FERModel
from .dataset import get_data_loaders

def evaluate_model(data_dir, model_path="model_no_disgust.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # We need to exclude disgust to match the 6-class model
    # Note: If the model file is for 7 classes, we should technically check, 
    # but the user context implies we are evaluating the "no disgust" model.
    train_loader, test_loader = get_data_loaders(data_dir, batch_size=64, excluded_classes=['disgust'])
    
    if not test_loader:
        print("Test loader is empty. Check data directory.")
        return

    classes = train_loader.dataset.classes
    num_classes = len(classes)
    print(f"Evaluating on {num_classes} classes: {classes}")

    # Initialize model
    model = FERModel(num_classes=num_classes).to(device)
    
    full_model_path = os.path.join(os.path.dirname(__file__), model_path)
    if os.path.exists(full_model_path):
        print(f"Loading weights from {full_model_path}...")
        try:
            model.load_state_dict(torch.load(full_model_path, map_location=device))
        except Exception as e:
            print(f"Error loading model: {e}")
            return
    else:
        print(f"Model file not found at {full_model_path}")
        return

    print("Starting evaluation...")
    model.eval()
    correct = 0
    total = 0
    class_correct = list(0. for i in range(num_classes))
    class_total = list(0. for i in range(num_classes))
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    overall_acc = 100 * correct / total
    print(f'\nOverall Accuracy: {overall_acc:.2f}%')
    
    print("\nPer-Class Accuracy:")
    for i in range(num_classes):
        if class_total[i] > 0:
            print(f'{classes[i]}: {100 * class_correct[i] / class_total[i]:.2f}%')
        else:
            print(f'{classes[i]}: N/A')

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "data")
    evaluate_model(data_dir)
