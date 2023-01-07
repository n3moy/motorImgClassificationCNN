import os
import yaml
import logging
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from yaml.loader import SafeLoader
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

    def stratified_split(self, dataset : torch.utils.data.Dataset, random_state=None):
        import random 
        from collections import defaultdict

        fraction = self.config['valid_ratio']
        labels = [torch.argmax(i[1]).item() for i in dataset]

        if random_state: 
            random.seed(random_state)
        indices_per_label = defaultdict(list)
        for index, label in enumerate(labels):
            indices_per_label[label].append(index)
        first_set_inds, second_set_inds = list(), list()
        for label, indices in indices_per_label.items():
            n_samples_for_label = round(len(indices) * fraction)
            random_indices_sample = random.sample(indices, n_samples_for_label)
            first_set_inds.extend(random_indices_sample)
            second_set_inds.extend(set(indices) - set(random_indices_sample))

        return second_set_inds, first_set_inds 

    def prepare(self, mode):
        if mode == 'train':
            tmp_dataset = MotorDataset(
                self.config['folder_path']
            )
            self.get_mean_std(tmp_dataset)
            del tmp_dataset

        tf = transforms.Compose([
            transforms.Scale((240, 240)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.config['mean'], std=self.config['std'])
        ])
        
        if mode == 'train':
            from torch.utils.data.sampler import SubsetRandomSampler
            from pathlib import Path
            
            folder_path = self.config['folder_path']
            dataset = MotorDataset(folder_path, transform=tf)
            logging.info(f'Training on {len(dataset)} samples from\n{folder_path}')
            batch_size = self.config['batch_size']
            train_inds, val_inds = self.stratified_split(dataset)
            train_sampler, val_sampler = SubsetRandomSampler(train_inds), SubsetRandomSampler(val_inds)
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

            for config_name in self.config['update_mean_std_configs']:
                config_path = Path(self.config['config_folder']) / config_name
                config_ = yaml.load(open(config_path), Loader=SafeLoader)
                config_['mean'] = self.config['mean'].numpy().tolist()
                config_['std'] = self.config['std'].numpy().tolist()
                yaml.dump(config_, open(config_path, 'w'))
            
            return train_loader, val_loader

        if mode == 'test':
            folder_path = self.config['folder_path']
            dataset = MotorDataset(folder_path, transform=tf)
            logging.info(f'Testing on {len(dataset)} samples from\n{folder_path}')
            batch_size = self.config['batch_size']
            loader = torch.utils.data.DataLoader(
                dataset, 
                batch_size=batch_size
            )
            return loader
        
        if mode == 'inference':
            instance = Image.open(os.path.join(self.config['image_path']))
            instance = tf(instance)
            instance = instance.unsqueeze(0)
            return instance

