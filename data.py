"""Dataset, трансформации и загрузка данных."""
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

from config import (
    TRAIN_DIR, TEST_DIR, CLASSES, CLASS2IDX,
    IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD,
    BATCH_SIZE, TEST_BATCH_SIZE, VAL_SPLIT, SEED,
)


class PlatesDataset(Dataset):
    def __init__(self, paths, labels=None, transform=None):
        self.paths = list(paths)
        self.labels = list(labels) if labels is not None else None
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        if self.labels is not None:
            return img, self.labels[idx]
        return img


def build_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return train_tf, val_tf


def collect_train_paths():
    paths, labels = [], []
    for cls in CLASSES:
        for p in sorted((TRAIN_DIR / cls).glob('*.jpg')):
            paths.append(p)
            labels.append(CLASS2IDX[cls])
    return np.array(paths), np.array(labels)


def collect_test_paths():
    return sorted(TEST_DIR.glob('*.jpg'))


def make_loaders(image_paths, labels, test_paths, train_tf, val_tf):
    train_idx, val_idx = train_test_split(
        np.arange(len(image_paths)),
        test_size=VAL_SPLIT,
        stratify=labels,
        random_state=SEED,
    )

    train_ds = PlatesDataset(image_paths[train_idx], labels[train_idx], train_tf)
    val_ds = PlatesDataset(image_paths[val_idx], labels[val_idx], val_tf)
    test_ds = PlatesDataset(test_paths, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader, train_idx, val_idx
