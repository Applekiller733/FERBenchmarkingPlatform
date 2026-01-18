from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import os
from typing import List, Optional

class CustomImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None, excluded_classes: Optional[List[str]] = None):
        self.excluded_classes = set(excluded_classes) if excluded_classes else set()
        super().__init__(root, transform=transform)

    def find_classes(self, directory: str):
        classes, class_to_idx = super().find_classes(directory)
        if self.excluded_classes:
            classes = [c for c in classes if c not in self.excluded_classes]
            # Re-index classes to be continuous 0 to N-1
            class_to_idx = {c: i for i, c in enumerate(classes)}
        return classes, class_to_idx

def get_data_loaders(data_dir, batch_size=64, excluded_classes=None):
    """
    Creates DataLoaders for train and test sets.
    Expects data_dir to contain 'train' and 'test' subdirectories.
    Structure:
    data_dir/
        train/
            angry/
            ...
        test/
            angry/
            ...
    
    Args:
        data_dir (str): Path to data directory
        batch_size (int): Batch size
        excluded_classes (list): List of class names to exclude (e.g., ['disgust'])
    """
    
    # FER2013 images are 48x48 Grayscale
    data_transforms = {
        'train': transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)) 
        ]),
        'test': transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]),
    }

    image_datasets = {}
    dataloaders = {}
    
    for x in ['train', 'test']:
        path = os.path.join(data_dir, x)
        if not os.path.exists(path):
            print(f"Warning: Path {path} does not exist. Please create it and populate with images.")
            # Return empty if path doesn't exist to prevent crash during setup, 
            # but user needs to fix this for training.
            return None, None
            
        image_datasets[x] = CustomImageFolder(path, data_transforms[x], excluded_classes=excluded_classes)
        # num_workers=4 allows parallel data loading, pin_memory=True speeds up transfer to GPU
        # persistent_workers=True keeps workers alive between epochs (requires num_workers > 0)
        dataloaders[x] = DataLoader(image_datasets[x], batch_size=batch_size, shuffle=(x == 'train'), 
                                  num_workers=4, pin_memory=True, persistent_workers=True)

    return dataloaders['train'], dataloaders['test']
