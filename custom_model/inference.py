import torch
from torchvision import transforms
from PIL import Image
import io
import os
from .model import FERModel

class EmotionPredictor:
    def __init__(self, model_path="model.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = FERModel(num_classes=7).to(self.device)
        
        # Load weights
        full_model_path = os.path.join(os.path.dirname(__file__), model_path)
        if os.path.exists(full_model_path):
            self.model.load_state_dict(torch.load(full_model_path, map_location=self.device))
            self.model.eval()
            print(f"Model loaded from {full_model_path}")
        else:
            print(f"Warning: Model not found at {full_model_path}. Predictions will be random initialized.")

        # Classes usually for FER2013 (Verify correspondence with your dataset folder names)
        # Note: If using ImageFolder, the class index depends on alphabetical order of folders.
        # Ideally, we should save class_to_idx mapping during training.
        # Assuming standard FER2013 folders: angry, disgust, fear, happy, neutral, sad, surprise
        self.classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

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
