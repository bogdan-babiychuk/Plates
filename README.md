# Классификация номерных знаков: чистый vs грязный

Бинарная классификация изображений автомобильных номерных знаков на классы `cleaned` / `dirty`. Решение основано на Transfer Learning поверх EfficientNet-B0 с двухфазной стратегией обучения.

## Структура проекта

```
plates/
├── main.py            # Точка входа: EDA -> обучение -> оценка -> предсказания
├── config.py          # Пути, гиперпараметры, device
├── data.py            # PlatesDataset, трансформации, DataLoader'ы
├── model.py           # EfficientNet-B0 + кастомная голова, freeze/unfreeze
├── train.py           # train_epoch, val_epoch, фазы 1 и 2
├── evaluate.py        # Метрики, classification_report, предсказания с картинками
├── predict.py         # Инференс на тесте + сохранение CSV
├── visualize.py       # EDA, кривые обучения, CM, ROC, анализ ошибок
├── report.md          # Отчёт по работе
├── task.md            # Постановка задачи
├── data/plates/
│   ├── train/
│   │   ├── cleaned/   # 20 изображений
│   │   └── dirty/     # 20 изображений
│   └── test/          # 744 изображения без меток
└── outputs/           # Создаётся при запуске: веса, графики, CSV
```

## Требования

- Python 3.9+
- PyTorch + torchvision
- numpy, pandas, scikit-learn
- matplotlib, seaborn, pillow

Установка:

```bash
pip install torch torchvision numpy pandas scikit-learn matplotlib seaborn pillow
```

## Запуск

Из директории проекта:

```bash
python main.py
```

Скрипт последовательно:

1. Собирает пути к train/test изображениям.
2. Сохраняет графики EDA в `outputs/` (`eda_all_samples.png`, `eda_image_stats.png`, `augmentation_examples.png`).
3. Делит train на train/val (80/20, стратифицированно).
4. Строит EfficientNet-B0 и обучает в две фазы:
   - **Фаза 1:** заморожен backbone, обучается голова (LR=1e-3, 30 эпох).
   - **Фаза 2:** размораживаются все слои, fine-tuning (LR backbone=1e-5, head=5e-5, 30 эпох).
5. Сохраняет лучшие веса в `outputs/best_model.pth`.
6. Считает метрики на валидации (Accuracy, Precision, Recall, F1, ROC-AUC), рисует CM и ROC.
7. Прогоняет тестовый набор и сохраняет `outputs/test_predictions.csv`.

## Конфигурация

Все ключевые параметры собраны в `config.py`:

| Параметр | Значение | Описание |
|---|---|---|
| `IMG_SIZE` | 224 | Размер входа EfficientNet-B0 |
| `BATCH_SIZE` | 8 | Размер тренировочного батча |
| `VAL_SPLIT` | 0.2 | Доля валидации |
| `EPOCHS_PHASE1` | 30 | Эпох в Фазе 1 |
| `EPOCHS_PHASE2` | 30 | Эпох в Фазе 2 |
| `LR_PHASE1` | 1e-3 | LR головы в Фазе 1 |
| `LR_BACKBONE_PHASE2` | 1e-5 | LR backbone в Фазе 2 |
| `LR_HEAD_PHASE2` | 5e-5 | LR головы в Фазе 2 |
| `DROPOUT` | 0.4 | Dropout перед классификатором |
| `WEIGHT_DECAY` | 1e-4 | L2-регуляризация |

## Артефакты в `outputs/`

- `best_model.pth` — веса лучшей по val accuracy модели.
- `test_predictions.csv` — `filename, prediction, label, prob_dirty, prob_clean`.
- Графики: `eda_*.png`, `augmentation_examples.png`, `learning_curves.png`, `confusion_matrix.png`, `roc_curve.png`, `error_analysis.png`, `prediction_confidence.png`, `test_predictions_vis.png`, `test_confidence_dist.png`.

## Воспроизводимость

`SEED=42` фиксируется в `torch` и `numpy`. Случайность в DataLoader/аугментациях детерминирована тем же seed. На GPU полная битовая воспроизводимость не гарантируется (cudnn-нондетерминизм).
