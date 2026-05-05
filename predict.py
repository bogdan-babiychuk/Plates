"""Предсказания на тестовом наборе."""
import numpy as np
import pandas as pd
import torch

from config import device, IDX2CLASS, PREDICTIONS_CSV, OUTPUT_DIR


@torch.no_grad()
def predict_test(model, loader):
    model.eval()
    all_preds, all_probs = [], []
    for images in loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)[:, 1]
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
    return np.array(all_preds), np.array(all_probs)


def save_predictions(test_paths, test_preds, test_probs):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        'filename': [p.name for p in test_paths],
        'prediction': [IDX2CLASS[p] for p in test_preds],
        'label': test_preds,
        'prob_dirty': test_probs.round(4),
        'prob_clean': (1 - test_probs).round(4),
    })
    df.to_csv(PREDICTIONS_CSV, index=False)
    print(f'Предсказания сохранены: {PREDICTIONS_CSV}')
    print(f'Чистых: {(df.label == 0).sum()}, Грязных: {(df.label == 1).sum()}')
    return df
