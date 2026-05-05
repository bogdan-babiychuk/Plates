"""Метрики, матрица ошибок, ROC-кривая, анализ ошибок."""
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
)

from config import device, IDX2CLASS
from train import val_epoch


def compute_metrics(val_true, val_probs, threshold=0.5):
    val_preds = (val_probs >= threshold).astype(int)
    metrics = {
        'accuracy': accuracy_score(val_true, val_preds),
        'precision': precision_score(val_true, val_preds, zero_division=0),
        'recall': recall_score(val_true, val_preds, zero_division=0),
        'f1': f1_score(val_true, val_preds, zero_division=0),
        'roc_auc': roc_auc_score(val_true, val_probs) if len(np.unique(val_true)) > 1 else float('nan'),
        'preds': val_preds,
    }
    return metrics


def print_metrics_report(metrics, val_true):
    print('=' * 50)
    print('МЕТРИКИ НА ВАЛИДАЦИОННОЙ ВЫБОРКЕ')
    print('=' * 50)
    print(f'Accuracy:  {metrics["accuracy"]:.4f}')
    print(f'Precision: {metrics["precision"]:.4f}')
    print(f'Recall:    {metrics["recall"]:.4f}')
    print(f'F1-score:  {metrics["f1"]:.4f}')
    print(f'ROC-AUC:   {metrics["roc_auc"]:.4f}')
    print()
    print(classification_report(val_true, metrics['preds'],
                                target_names=['Чистый', 'Грязный']))


def evaluate_model(model, val_loader):
    criterion = nn.CrossEntropyLoss()
    _, _, val_probs, val_true = val_epoch(model, val_loader, criterion)
    metrics = compute_metrics(val_true, val_probs)
    print_metrics_report(metrics, val_true)
    cm = confusion_matrix(val_true, metrics['preds'])
    return val_true, val_probs, metrics, cm


@torch.no_grad()
def get_predictions_with_images(model, paths, labels_list, transform):
    model.eval()
    results = []
    for path, true_lbl in zip(paths, labels_list):
        img_orig = Image.open(path).convert('RGB')
        img_t = transform(img_orig).unsqueeze(0).to(device)
        out = model(img_t)
        prob = torch.softmax(out, dim=1)[0, 1].item()
        pred = int(prob >= 0.5)
        results.append({
            'path': path, 'true': int(true_lbl), 'pred': pred,
            'prob_dirty': prob, 'correct': pred == int(true_lbl),
            'img': img_orig,
            'true_name': IDX2CLASS[int(true_lbl)],
            'pred_name': IDX2CLASS[pred],
        })
    return results
