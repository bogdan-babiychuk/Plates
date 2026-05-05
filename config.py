"""Константы и настройки проекта."""
from pathlib import Path
import torch

SEED = 42

DATA_DIR = Path('data/plates')
TRAIN_DIR = DATA_DIR / 'train'
TEST_DIR = DATA_DIR / 'test'

CLASSES = ['cleaned', 'dirty']
CLASS_RU = {'cleaned': 'Чистый', 'dirty': 'Грязный'}
CLASS2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2CLASS = {i: c for c, i in CLASS2IDX.items()}

IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

BATCH_SIZE = 8
TEST_BATCH_SIZE = 16
VAL_SPLIT = 0.2

EPOCHS_PHASE1 = 30
EPOCHS_PHASE2 = 30
LR_PHASE1 = 1e-3
LR_BACKBONE_PHASE2 = 1e-5
LR_HEAD_PHASE2 = 5e-5
WEIGHT_DECAY = 1e-4
DROPOUT = 0.4

OUTPUT_DIR = Path('outputs')
MODEL_PATH = OUTPUT_DIR / 'best_model.pth'
PREDICTIONS_CSV = OUTPUT_DIR / 'test_predictions.csv'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
