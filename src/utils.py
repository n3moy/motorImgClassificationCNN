import yaml
import logging
from yaml.loader import SafeLoader

import torch
import torch.nn as nn

from models.trainer import Trainer
from models.model import CNN
from data.img_handler import ImgHandler


def train_model(config_path):
    config = yaml.load(open(config_path), Loader=SafeLoader)
    handler = ImgHandler(config)
    train_loader, val_loader = handler.prepare(mode='train')
    model_cnn = CNN(img_size=None, out_size=11)
    # use_cuda = torch.cuda.is_available()
    # device = torch.device('cuda:0' if use_cuda else 'cpu')
    device = 'cpu'
    trainer = Trainer(model_cnn, device=device)
    n_epochs = config['n_epochs']
    lr = config['learning_rate']
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_cnn.parameters(), lr=lr)
    loss_history, best_model = trainer.train_CNN(
                                    train_loader, 
                                    val_loader, 
                                    loss, 
                                    optimizer, 
                                    n_epochs
                                )
    print(loss_history)


