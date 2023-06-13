
import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset

class VLImageDataset(Dataset):
    def __init__(self, image_features, text_features):
        self.image_features = image_features
        self.text_features = text_features


    def __len__(self):
        return self.text_features.shape[0]

    def __getitem__(self, idx):
        image = self.image_features[idx]
        text = self.text_features[idx]
        return image, text