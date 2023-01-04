import yaml
import logging
from yaml.loader import SafeLoader
from pathlib import Path

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
    loss_history, model = trainer.train_CNN(
                                    train_loader, 
                                    val_loader, 
                                    loss, 
                                    optimizer, 
                                    n_epochs
                                )
    
    if 'save_model_path' in config:
        save_path = Path(config['save_model_path'])
        torch.save(model.state_dict(), save_path)
        logging.info(f'Trained model is saved to : {save_path}')
    else:
        error_string = 'No path in config to save model'
        logging.error(error_string)
        raise NotImplementedError(error_string)

    results_filename = 'loss_hist.yaml'
    results_path = Path(__file__).parent.parent / 'logs' / 'train' / results_filename
    yaml.dump(loss_history, open(results_path, 'w'), sort_keys=False)
    logging.info(f'Loss history is saved to {results_path}')


def test_model(config_path):
    from models.evaluation import evaluate_model, get_quality_metrics

    config = yaml.load(open(config_path), Loader=SafeLoader)
    handler = ImgHandler(config)
    test_loader = handler.prepare(mode='test')
    model_cnn = CNN(img_size=None, out_size=11)
    model_cnn.load_state_dict(torch.load(config['model_path']))
    preds, ground_truth = evaluate_model(model_cnn, test_loader)
    metrics = get_quality_metrics(preds, ground_truth)
    yaml.dump(metrics, open(config['output_path'], 'w'), sort_keys=False)

