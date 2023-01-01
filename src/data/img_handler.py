import os
import yaml
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image


class MotorDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.transform = transform
        self.root_dir = folder
        self.data = os.listdir(folder)

        self.mapping_y = [
            'A&B50', 
            'A&C&B10', 
            'A&C&B30', 
            'A&C10', 
            'A&C30',
            'A10',
            'A30',
            'A50',
            'Fan',
            'Noload',
            'Rotor-0' 
        ]

        self.y_0 = torch.zeros(len(self.mapping_y)) 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_id = self.data[index]
        img = Image.open(os.path.join(self.root_dir, img_id))
        # I take name of file as target
        metric = img_id.split('.')[0].split('_')[0]
        y = self.y_0.clone()
        y[self.mapping_y.index(metric)] = 1
        if self.transform:
            img = self.transform(img)

        return img, y, img_id


class ImgHandler():
    def __init__(self, config):
        self.config = config

    def get_mean_std(self, dataset):
        tensor_transform = transforms.ToTensor()
        mean = 0
        var = 0
        for ix, (img, _, _) in enumerate(dataset, start=1):
            img_tensor = tensor_transform(img)
            mean += img_tensor.mean([1, 2])
            var += img_tensor.var([1, 2])
        mean = mean / ix
        var = var / ix
        std = torch.sqrt(var)
        self.config['mean'] = mean
        self.config['std'] = std

    def prepare(self, mode):
        if mode == 'train':
            tmp_dataset = MotorDataset(
                self.config['folder_path']
            )
            self.get_mean_std(tmp_dataset)
            del tmp_dataset

        dataset = MotorDataset(
            self.config['folder_path'], 
            transform=transforms.Compose([
                transforms.Scale((240, 240)),
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
            train_ind, val_inds = indicies[val_split:], indicies[:val_split]
            train_sampler, val_sampler = SubsetRandomSampler(train_ind), SubsetRandomSampler(val_inds)
            train_loader = torch.utils.data.DataLoader(
                dataset, 
                batch_size=batch_size, 
                sampler=train_sampler
            )
            val_loader = torch.utils.data.DataLoader(
                dataset, 
                batch_size=batch_size, 
                sampler=val_sampler
            )

            if not 'mean' in self.config or not 'std' in self.config:
                yaml.dump(self.config, self.config['my_path'])

            return train_loader, val_loader

        if mode == 'test':
            pass


