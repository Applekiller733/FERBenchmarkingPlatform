import torch
from torchvision import transforms
from PIL import Image
import io
import os
from .model import FERModel

class EmotionPredictor:
    def __init__(self, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        base_dir = os.path.dirname(__file__)
        
        # Determine model path and classes
        if model_path is None:
            no_disgust_path = os.path.join(base_dir, "model_no_disgust.pth")
            if os.path.exists(no_disgust_path):
                model_path = "model_no_disgust.pth"
                # Alphabetical order without 'disgust'
                self.classes = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']
                print(f"Auto-selected model: {model_path} (6 classes)")
            else:
                model_path = "model.pth"
                self.classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
                print(f"Using default model: {model_path} (7 classes)")
        else:
             # If path provided, guess classes based on filename or default to 7 (risky but simple)
             # Better to separate class definition if fully custom. For now, assume 7 unless known name.
             if "no_disgust" in model_path:
                 self.classes = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']
             else:
                 self.classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

        # Initialize model with correct num_classes
        self.model = FERModel(num_classes=len(self.classes)).to(self.device)
        
        # Load weights
        full_model_path = os.path.join(base_dir, model_path)
        if os.path.exists(full_model_path):
            try:
                self.model.load_state_dict(torch.load(full_model_path, map_location=self.device))
                self.model.eval()
                print(f"Model loaded successfully from {full_model_path}")
            except Exception as e:
                print(f"Error loading weights: {e}")
        else:
            print(f"Warning: Model not found at {full_model_path}. Predictions will be random initialized.")

        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def predict(self, image_bytes):
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB') # Ensure it opens even if originally grayscale
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                
                top_prob, top_class = probs.topk(1, dim=1)
                
                predicted_idx = top_class.item()
                confidence = top_prob.item()
                
                return {
                    "top_emotion": self.classes[predicted_idx] if predicted_idx < len(self.classes) else str(predicted_idx),
                    "confidence": confidence,
                    "predictions": {self.classes[i]: probs[0][i].item() for i in range(len(self.classes))}
                }
        except Exception as e:
            print(f"Prediction error: {e}")
            return None
