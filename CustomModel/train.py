import torch
import torch.nn as nn
import torch.optim as optim
import os
from .model import FERModel
from .dataset import get_data_loaders

def train_model(data_dir, num_epochs=15, batch_size=64, learning_rate=0.001, model_path="model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model
    model = FERModel(num_classes=7).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Data loaders
    train_loader, test_loader = get_data_loaders(data_dir, batch_size)
    
    if not train_loader:
        print("Training cannot start. Please check dataset path.")
        return

    print("Starting training...")
    from tqdm import tqdm
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        # Wrap train_loader with tqdm for progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
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

if __name__ == "__main__":
    # Default data directory assumed to be in CustomModel/data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "data")
    train_model(data_dir)
