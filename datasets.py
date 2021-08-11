from torch.utils.data import Dataset
import torch
from pathlib import Path
from torchvision.io import read_image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

class PACSDataset(Dataset):
    # Required
    def __init__(self, file_names, transform=None):
        """
        Args:
            file_names (list): list of file names for images of the dataset (train or test split).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.label_mapping = {"dog": 0,
                              "elephant": 1, 
                              "giraffe": 2, 
                              "guitar": 3, 
                              "horse": 4, 
                              "house": 5, 
                              "person": 6
                             }
    
        self.file_names = file_names
        self.transform = transform

    # Required   
    def __len__(self):
        return len(self.file_names)
    
    def get_label(self, file_name):
        image_path = Path(file_name)
        return image_path.parent.name
    
    # Required
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.file_names[idx]
        image = read_image(img_name)
        image = image.type(torch.FloatTensor)/255
        label = self.get_label(img_name)
        label = self.label_mapping[label]

        if self.transform:
            image = self.transform(image)
                    
        return (image, label)