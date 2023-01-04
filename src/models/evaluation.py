import torch
import numpy as np


def evaluate_model(model, loader):
    """
    Computes predictions and ground truth labels for the indices of the dataset
    
    Returns: 
    predictions: np array of ints - model predictions
    grount_truth: np array of ints - actual labels of the dataset
    """
    model.eval()
    predictions = []
    gts = []
    for x, y, _ in loader:
        pred = torch.argmax(model(x), 1).tolist()
        ground_truth = torch.argmax(y, 1).tolist()
        predictions.extend(pred)
        gts.extend(ground_truth)

    return predictions, gts


def get_quality_metrics(preds, ground_truth):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    result = {
        'accuracy': float(accuracy_score(ground_truth, preds)),
        'precision': float(precision_score(ground_truth, preds, average='macro')),
        'recall': float(recall_score(ground_truth, preds, average='macro')),
        'f1_score': float(f1_score(ground_truth, preds, average='macro'))
    }
    return result