import os
import yaml
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset, SubsetRandomSampler
from PIL import Image


class MotorDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.transform = transform
        self.root_dir = folder
        self.data = os.listdir(folder)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img_id = self.data[index]
        img = Image.open(os.path.join(self.root_dir, img_id))
        # I take name of file as target
        y = img_id.split('.')[0].split('_')[0]

        if self.transform:
            img = self.transform(img)

        return img, y, img_id


class ImgHandler():
    def __init__(self, config_path):
        self.config = yaml.load(config_path)

    def get_mean_std(self, dataset):
        tensor_transform = transforms.ToTensor()
        mean = 0
        var = 0
        for ix, img_ in enumerate(dataset, start=1):
            img = img_[0]
            img_tensor = tensor_transform(img)
            mean += img_tensor.mean([1, 2])
            var += img_tensor.var([1, 2])
        mean = mean / ix
        var = var / ix
        std = torch.sqrt(var)
        self.config['mean'] = mean
        self.config['std'] = std

    def prepare(self, folder_path, mode):
        if mode == 'train':
            self.get_mean_std()

        dataset = MotorDataset(
            folder_path, 
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=self.config['mean'], std=self.config['std'])
            ])
        )
        if mode == 'train':
            from torch.utils.data.sampler import SubsetRandomSampler

            batch_size = self.config['batch_size']
            valid_ratio = self.config['valid_ratio']
            val_split = int(len(dataset) * valid_ratio)
            indicies = torch.arange(len(dataset))
            np.random.shuffle(indicies)

        pass


