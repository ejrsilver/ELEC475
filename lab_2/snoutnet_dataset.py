import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset

class SnoutNetDataset(Dataset):
    def __init__(self, annotations_file, imgs_dir, transform=None, target_transform=None):
        self.annotations = pd.read_csv(annotations_file)
        self.imgs_dir = imgs_dir
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        #annotations csv is in form <img name>,"<location>"
        img_path = os.path.join(self.imgs_dir, self.annotations.iloc[idx, 0])
        img = read_image(img_path)
        label = self.annotations.iloc[idx,1]

        if self.transform:
            img = self.transform(img)
        
        if self.target_transform:
            label = self.target_transform(label)

        return img, label