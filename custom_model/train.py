import torch
import torch.nn as nn
import torch.optim as optim
import os
from .model import FERModel
from .dataset import get_data_loaders

def train_model(data_dir, num_epochs=15, batch_size=64, learning_rate=0.001, model_path="model.pth", excluded_classes=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data loaders
    # We load data first to know num_classes
    train_loader, test_loader = get_data_loaders(data_dir, batch_size, excluded_classes=excluded_classes)
    
    if not train_loader:
        print("Training cannot start. Please check dataset path.")
        return

    # Determine number of classes dynamically
    classes = train_loader.dataset.classes
    num_classes = len(classes)
    print(f"Training on {num_classes} classes: {classes}")

    # Initialize model
    model = FERModel(num_classes=num_classes).to(device)
    
    full_model_path = os.path.join(os.path.dirname(__file__), model_path)
    if os.path.exists(full_model_path):
        print(f"Loading existing weights from {full_model_path}...")
        try:
            # map_location ensures it loads correctly even if trained on GPU and running on CPU
            state_dict = torch.load(full_model_path, map_location=device)
            
            # Check if the stored model has compatible final layer
            # If we are changing from 7 classes to 6, the final layer size will mismatch.
            # We can load the partial state dict.
            model_dict = model.state_dict()
            
            # Filter out unnecessary keys
            pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
            
            if len(pretrained_dict) != len(model_dict):
                print("Warning: Some weights could not be loaded due to architecture mismatch (likely different number of classes).")
                print(f"Loaded {len(pretrained_dict)}/{len(model_dict)} layers.")
            
            # Overwrite entries in the existing state dict
            model_dict.update(pretrained_dict) 
            # Load the new state dict
            model.load_state_dict(model_dict)
        except Exception as e:
            print(f"Error loading model: {e}. Starting from scratch.")
    else:
        print("No existing model found. Starting from scratch.")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("Starting training...")
    from tqdm import tqdm
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        # Wrap train_loader with tqdm for progress bar
        # persistent_workers=True in dataset.py helps, but we just iterate here
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Update progress bar description with current loss
            pbar.set_postfix({'loss': running_loss / (pbar.n + 1)})
            
        print(f"Epoch [{epoch+1}/{num_epochs}] Completed. Avg Loss: {running_loss/len(train_loader):.4f}")
        
    # Save model
    save_path = os.path.join(os.path.dirname(__file__), model_path)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    # ==========================
    # Validation / Evaluation
    # ==========================
    print("\nRunning Evaluation on Test Set...")
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

    print(f'\nOverall Accuracy of the model on the test images: {100 * correct / total:.2f}%')
    
    print("\nPer-Class Accuracy:")
    for i in range(num_classes):
        if class_total[i] > 0:
            print(f'Accuracy of {classes[i]:<10} : {100 * class_correct[i] / class_total[i]:.2f}%')
        else:
            print(f'Accuracy of {classes[i]:<10} : N/A (No samples)')

if __name__ == "__main__":
    # Default data directory assumed to be in CustomModel/data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "data")
    
    # User requested training without 'disgust' class
    train_model(data_dir, excluded_classes=['disgust'], model_path="model_no_disgust.pth")
