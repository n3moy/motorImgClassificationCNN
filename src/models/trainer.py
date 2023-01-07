import logging
from tqdm import tqdm
from copy import deepcopy

import torch


class Trainer():
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.type(torch.cuda.FloatTensor)
        self.model.to(self.device)

    def train_CNN(self, train_loader, val_loader, loss, optimizer, n_epochs):
        loss_history = {
            'train': [],
            'validation': []
        }
        best_loss = 10000
        best_model = None
        best_epoch = 0
        logging.info('Starting model training process')
        for i_epoch in range(n_epochs):
            print(f'# {i_epoch} / {n_epochs} epoch')
            self.model.train()
            train_loss = 0
            valid_loss = 0

            for x, y, _ in tqdm(train_loader):
                x_device = x.to(self.device)
                y_device = y.to(self.device)
                preds = self.model(x_device)
                loss_value = loss(preds, y_device)
                train_loss += loss_value.item()
                optimizer.zero_grad()
                loss_value.backward()
                optimizer.step()

            train_loss = train_loss / len(train_loader)
            print(f'Train loss : {train_loss}')
            loss_history['train'].append(train_loss)

            self.model.eval()
            with torch.no_grad():
                for x, y, _ in tqdm(val_loader):
                    x_device = x.to(self.device)
                    y_device = y.to(self.device)
                    preds = self.model(x_device)
                    loss_value = loss(preds, y_device)
                    valid_loss += loss_value.item()
            valid_loss = valid_loss / len(val_loader)
            print(f'Validation loss : {valid_loss}')
            loss_history['validation'].append(valid_loss)

            if valid_loss < best_loss:
                best_loss = valid_loss
                best_model = deepcopy(self.model)
                best_epoch = i_epoch
            logging.info(f'Epoch {i_epoch}, best loss at {best_epoch} : {best_loss}\nCurrent loss : train - {train_loss} ; val = {valid_loss}')
        logging.info('Trainig is done')

        return loss_history, best_model
