import os, re
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
from PIL import Image
import torch

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
        img_nonformat = Image.open(img_path)
        img = img_nonformat.convert('RGB').resize((227,227))
        
        #need to get coordinates
        file_coords = self.annotations.iloc[idx,1]
        coords = re.findall(r"\((\d+)\,\s(\d+)\)", file_coords)[0]
        coords = [int(x) for x in coords]

        #scale coordinates
        scale_x = 227 / img_nonformat.size[0]
        scale_y = 227 / img_nonformat.size[1]

        x = int(coords[0] * scale_x)
        y = int(coords[1] * scale_y)
        
        label = torch.tensor([x,y])
        
        if self.transform:
            img = self.transform(img)
        
        if self.target_transform:
            label = self.target_transform(label)

        return img, label