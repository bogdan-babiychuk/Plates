"""Циклы обучения и валидации с двухфазной стратегией."""
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import (
    device, EPOCHS_PHASE1, EPOCHS_PHASE2,
    LR_PHASE1, LR_BACKBONE_PHASE2, LR_HEAD_PHASE2, WEIGHT_DECAY,
)
from model import freeze_backbone, unfreeze_all, count_params


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def val_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_probs, all_labels = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        probs = torch.softmax(outputs, dim=1)[:, 1]
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return total_loss / total, correct / total, np.array(all_probs), np.array(all_labels)


def run_training(model, train_loader, val_loader, optimizer, scheduler,
                 criterion, num_epochs, phase_name=''):
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_acc, best_state = 0.0, None

    for epoch in range(1, num_epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion)
        vl_loss, vl_acc, _, _ = val_epoch(model, val_loader, criterion)
        if scheduler:
            scheduler.step()

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(vl_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(vl_acc)

        if vl_acc > best_acc:
            best_acc = vl_acc
            best_state = deepcopy(model.state_dict())

        if epoch % 10 == 0 or epoch == 1:
            print(f'[{phase_name}] Epoch {epoch:3d}/{num_epochs} | '
                  f'Loss: {tr_loss:.4f}/{vl_loss:.4f} | '
                  f'Acc: {tr_acc:.3f}/{vl_acc:.3f}')

    print(f'[{phase_name}] Лучшая val accuracy: {best_acc:.4f}')
    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def train_phase1(model, train_loader, val_loader):
    freeze_backbone(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_PHASE1, weight_decay=WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    print('=== ФАЗА 1: Обучение головы (backbone заморожен) ===')
    total, trainable = count_params(model)
    print(f'Обучаемых параметров: {trainable:,} из {total:,}')
    return run_training(model, train_loader, val_loader, optimizer, scheduler,
                        criterion, EPOCHS_PHASE1, phase_name='Фаза 1')


def train_phase2(model, train_loader, val_loader):
    unfreeze_all(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([
        {'params': model.features.parameters(), 'lr': LR_BACKBONE_PHASE2},
        {'params': model.classifier.parameters(), 'lr': LR_HEAD_PHASE2},
    ], weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_PHASE2, eta_min=1e-7)

    print('=== ФАЗА 2: Fine-tuning (все слои разморожены) ===')
    total, trainable = count_params(model)
    print(f'Обучаемых параметров: {trainable:,} из {total:,}')
    return run_training(model, train_loader, val_loader, optimizer, scheduler,
                        criterion, EPOCHS_PHASE2, phase_name='Фаза 2')
