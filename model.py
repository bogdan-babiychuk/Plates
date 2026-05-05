"""Архитектура модели на базе EfficientNet-B0 (Transfer Learning)."""
import torch.nn as nn
from torchvision import models

from config import device, DROPOUT


def build_model(num_classes=2, dropout=DROPOUT):
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    return model.to(device)


def freeze_backbone(model):
    for p in model.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True


def unfreeze_all(model):
    for p in model.parameters():
        p.requires_grad = True


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
