from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import os

def get_data_loaders(data_dir, batch_size=64):
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
            
        image_datasets[x] = datasets.ImageFolder(path, data_transforms[x])
        dataloaders[x] = DataLoader(image_datasets[x], batch_size=batch_size, shuffle=(x == 'train'), num_workers=0) # num_workers=0 for Windows compatibility often helps

    return dataloaders['train'], dataloaders['test']
