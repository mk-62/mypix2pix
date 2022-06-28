import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import torchvision.transforms as tt

class PairDataset(Dataset):
    def __init__(self, path, target_size = 256):
        stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        self.transform = tt.Compose([tt.ConvertImageDtype(torch.float),
                            tt.Resize(target_size), tt.Normalize(*stats)])
        self.image_path_list = []        
        with os.scandir(path) as it:
          for entry in it:
                if not entry.name.startswith('.') and entry.is_file() and entry.name.endswith('.jpg'):
                    self.image_path_list.append(os.path.join(path,entry.name))
        self.image_path_list.sort()
    def __len__(self):
        return len(self.image_path_list)
    def __getitem__(self, i):
        image = read_image(self.image_path_list[i])
        pack = self.transform(torch.stack((image[:, :, :image.shape[2]//2],
                                           image[:, :, image.shape[2]//2:])))
        return { 'outline' : pack[0], 'solid' : pack[1] }

class TrainDataset(Dataset):
    def __init__(self, path, crop_size = 224, target_size = 256):
        stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        self.transform = tt.Compose([tt.ConvertImageDtype(torch.float), 
          tt.RandomCrop(crop_size), tt.Resize(target_size), tt.RandomHorizontalFlip(0.5), tt.Normalize(*stats)])
        self.image_path_list = []
        with os.scandir(path) as it:
          for entry in it:
                if not entry.name.startswith('.') and entry.is_file() and entry.name.endswith('.jpg'):
                    self.image_path_list.append(os.path.join(path,entry.name))
        self.image_path_list.sort()
    def __len__(self):
        return len(self.image_path_list)
    def __getitem__(self, i):
        image = read_image(self.image_path_list[i])
        pack = self.transform(torch.stack((image[:, :, :image.shape[2]//2],
                                           image[:, :, image.shape[2]//2:])))
        return { 'outline' : pack[0], 'solid' : pack[1] }