"""Главный скрипт: EDA → обучение → оценка → предсказания."""
import warnings

import numpy as np
import torch

from config import (
    SEED, device, CLASSES, CLASS_RU, CLASS2IDX,
    OUTPUT_DIR, MODEL_PATH,
)
from data import build_transforms, collect_train_paths, collect_test_paths, make_loaders
from model import build_model
from train import train_phase1, train_phase2
from evaluate import evaluate_model, get_predictions_with_images
from predict import predict_test, save_predictions
from visualize import (
    setup_matplotlib, plot_all_samples, plot_image_stats, plot_augmentation,
    plot_learning_curves, plot_confusion_matrix, plot_roc,
    plot_errors_and_confidence, plot_test_predictions,
)

warnings.filterwarnings('ignore')


def set_seed(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)


def main():
    set_seed()
    setup_matplotlib()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f'Устройство: {device}')
    print(f'PyTorch: {torch.__version__}\n')

    # 1. Данные
    image_paths, labels = collect_train_paths()
    test_paths = collect_test_paths()
    print(f'Train: {len(image_paths)} | Test: {len(test_paths)}')
    for cls in CLASSES:
        n = (labels == CLASS2IDX[cls]).sum()
        print(f'  {CLASS_RU[cls]}: {n}')

    train_tf, val_tf = build_transforms()

    # 2. EDA
    plot_all_samples(image_paths, labels)
    plot_image_stats(image_paths, labels)
    plot_augmentation(image_paths, labels, train_tf)

    # 3. Загрузчики
    train_loader, val_loader, test_loader, train_idx, val_idx = make_loaders(
        image_paths, labels, test_paths, train_tf, val_tf,
    )

    # 4. Модель + обучение
    model = build_model()
    history_p1 = train_phase1(model, train_loader, val_loader)
    history_p2 = train_phase2(model, train_loader, val_loader)

    torch.save(model.state_dict(), MODEL_PATH)
    print(f'\nМодель сохранена: {MODEL_PATH}')

    plot_learning_curves(history_p1, history_p2)

    # 5. Оценка
    val_true, val_probs, metrics, cm = evaluate_model(model, val_loader)
    plot_confusion_matrix(cm)
    plot_roc(val_true, val_probs)

    results = get_predictions_with_images(model, image_paths[val_idx], labels[val_idx], val_tf)
    plot_errors_and_confidence(results)

    # 6. Тест
    test_preds, test_probs = predict_test(model, test_loader)
    df = save_predictions(test_paths, test_preds, test_probs)
    plot_test_predictions(df, test_probs, test_preds)

    print('\nГотово.')


if __name__ == '__main__':
    main()
