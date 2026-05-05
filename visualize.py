"""EDA, кривые обучения, матрица ошибок, ROC, анализ ошибок."""
import numpy as np
import seaborn as sns
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from PIL import Image
from sklearn.metrics import roc_curve, auc

from config import (
    CLASSES, CLASS_RU, CLASS2IDX, IDX2CLASS,
    IMAGENET_MEAN, IMAGENET_STD, OUTPUT_DIR, TEST_DIR,
)


def setup_matplotlib():
    plt.rcParams['figure.dpi'] = 120
    plt.rcParams['font.family'] = 'DejaVu Sans'
    sns.set_style('whitegrid')


def _save(fig, name):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / name, bbox_inches='tight', dpi=120)


def plot_all_samples(image_paths, labels):
    fig, axes = plt.subplots(4, 10, figsize=(22, 9))
    fig.suptitle('Обучающая выборка: все 40 изображений', fontsize=14, fontweight='bold')
    colors = {'cleaned': '#2ecc71', 'dirty': '#e74c3c'}

    for cls_idx, cls in enumerate(CLASSES):
        cls_paths = image_paths[labels == cls_idx]
        for i, img_path in enumerate(cls_paths):
            row = cls_idx * 2 + i // 10
            col = i % 10
            ax = axes[row, col]
            ax.imshow(Image.open(img_path))
            ax.axis('off')
            for spine in ax.spines.values():
                spine.set_edgecolor(colors[cls])
                spine.set_linewidth(2)
                spine.set_visible(True)

    plt.tight_layout()
    _save(fig, 'eda_all_samples.png')
    plt.close(fig)


def plot_image_stats(image_paths, labels):
    widths, heights, means = [], [], []
    for p in image_paths:
        img = np.array(Image.open(p).convert('RGB'))
        heights.append(img.shape[0])
        widths.append(img.shape[1])
        means.append(img.mean() / 255.0)
    means_arr = np.array(means)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Статистика изображений', fontsize=13, fontweight='bold')

    cls_counts = [int((labels == 0).sum()), int((labels == 1).sum())]
    bars = axes[0].bar(['Чистый', 'Грязный'], cls_counts,
                       color=['#2ecc71', '#e74c3c'], edgecolor='black', linewidth=0.8)
    for bar, cnt in zip(bars, cls_counts):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                     str(cnt), ha='center', fontweight='bold')
    axes[0].set_title('Баланс классов')
    axes[0].set_ylabel('Количество')

    axes[1].scatter(widths, heights,
                    c=['#2ecc71' if l == 0 else '#e74c3c' for l in labels],
                    alpha=0.8, s=60, edgecolors='black', linewidth=0.5)
    axes[1].set_title('Размеры (W×H)')
    axes[1].set_xlabel('Ширина, px')
    axes[1].set_ylabel('Высота, px')
    axes[1].legend(handles=[
        mpatches.Patch(color='#2ecc71', label='Чистый'),
        mpatches.Patch(color='#e74c3c', label='Грязный'),
    ])

    axes[2].hist(means_arr[labels == 0], bins=8, alpha=0.7, color='#2ecc71', label='Чистый', edgecolor='black')
    axes[2].hist(means_arr[labels == 1], bins=8, alpha=0.7, color='#e74c3c', label='Грязный', edgecolor='black')
    axes[2].set_title('Средняя яркость')
    axes[2].set_xlabel('Средняя яркость (0-1)')
    axes[2].legend()

    plt.tight_layout()
    _save(fig, 'eda_image_stats.png')
    plt.close(fig)

    print(f'Размеры: от {min(widths)}x{min(heights)} до {max(widths)}x{max(heights)} px')
    print(f'Яркость — Чистые: {means_arr[labels==0].mean():.3f}, '
          f'Грязные: {means_arr[labels==1].mean():.3f}')


def plot_augmentation(image_paths, labels, train_transform):
    mean_t = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std_t = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    def denorm(t):
        return torch.clamp(t * std_t + mean_t, 0, 1)

    orig_clean = Image.open(image_paths[labels == 0][0]).convert('RGB')
    orig_dirty = Image.open(image_paths[labels == 1][0]).convert('RGB')

    fig, axes = plt.subplots(2, 7, figsize=(20, 6))
    fig.suptitle('Аугментация: что видит модель', fontsize=13, fontweight='bold')

    for row, (orig, name) in enumerate([(orig_clean, 'Чистый'), (orig_dirty, 'Грязный')]):
        axes[row, 0].imshow(orig)
        axes[row, 0].set_title('Оригинал', fontweight='bold')
        axes[row, 0].axis('off')
        axes[row, 0].set_ylabel(name, fontsize=11, fontweight='bold', rotation=90, labelpad=10)
        for col in range(1, 7):
            aug = denorm(train_transform(orig))
            axes[row, col].imshow(aug.permute(1, 2, 0).numpy())
            axes[row, col].set_title(f'Вариант {col}')
            axes[row, col].axis('off')

    plt.tight_layout()
    _save(fig, 'augmentation_examples.png')
    plt.close(fig)


def plot_learning_curves(history_p1, history_p2):
    full = {k: history_p1[k] + history_p2[k] for k in history_p1}
    n_epochs = len(full['train_loss'])
    boundary = len(history_p1['train_loss'])
    epochs = range(1, n_epochs + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Кривые обучения', fontsize=14, fontweight='bold')

    for ax, (key_tr, key_vl), title, ylabel in [
        (axes[0], ('train_loss', 'val_loss'), 'Loss', 'Loss'),
        (axes[1], ('train_acc', 'val_acc'), 'Accuracy', 'Accuracy'),
    ]:
        ax.plot(epochs, full[key_tr], label='Train', color='#3498db', linewidth=2)
        ax.plot(epochs, full[key_vl], label='Val', color='#e74c3c', linewidth=2)
        ax.axvline(boundary, color='gray', linestyle='--', alpha=0.7, label='Конец Фазы 1')
        ax.set_xlabel('Эпоха')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, 'learning_curves.png')
    plt.close(fig)

    print(f'Финальная Train Acc: {full["train_acc"][-1]:.4f}')
    print(f'Финальная Val Acc:   {full["val_acc"][-1]:.4f}')
    print(f'Лучшая Val Acc:      {max(full["val_acc"]):.4f}')


def plot_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Чистый', 'Грязный'],
                yticklabels=['Чистый', 'Грязный'],
                linewidths=0.5, linecolor='gray', ax=ax,
                annot_kws={'size': 16, 'fontweight': 'bold'})
    ax.set_xlabel('Предсказанный класс')
    ax.set_ylabel('Истинный класс')
    ax.set_title('Матрица ошибок', fontweight='bold')

    tn, fp, fn, tp = cm.ravel()
    print(f'TN={tn}  FP={fp}  FN={fn}  TP={tp}')

    plt.tight_layout()
    _save(fig, 'confusion_matrix.png')
    plt.close(fig)


def plot_roc(val_true, val_probs):
    if len(np.unique(val_true)) < 2:
        print('Недостаточно классов для ROC')
        return

    fpr, tpr, thresholds = roc_curve(val_true, val_probs)
    roc_auc = auc(fpr, tpr)
    j = tpr - fpr
    best_idx = int(np.argmax(j))

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color='#3498db', lw=2, label=f'ROC (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.6, label='Случайная модель')
    ax.scatter(fpr[best_idx], tpr[best_idx], color='#e74c3c', s=100, zorder=5,
               label=f'Опт. порог = {thresholds[best_idx]:.2f}')
    ax.fill_between(fpr, tpr, alpha=0.1, color='#3498db')
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.set_title('ROC-кривая', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, 'roc_curve.png')
    plt.close(fig)

    print(f'AUC = {roc_auc:.4f}, оптимальный порог = {thresholds[best_idx]:.3f}')


def plot_errors_and_confidence(results):
    errors = [r for r in results if not r['correct']]
    print(f'Валидация: {sum(r["correct"] for r in results)}/{len(results)} правильно, {len(errors)} ошибок')

    if errors:
        fig, axes = plt.subplots(1, len(errors), figsize=(5 * len(errors), 4))
        if len(errors) == 1:
            axes = [axes]
        fig.suptitle('Ошибочные предсказания', fontsize=13, fontweight='bold')
        for ax, err in zip(axes, errors):
            ax.imshow(err['img'])
            ax.set_title(
                f'Истина: {CLASS_RU[err["true_name"]]}\n'
                f'Предсказание: {CLASS_RU[err["pred_name"]]}\n'
                f'P(dirty)={err["prob_dirty"]:.3f}',
                color='#e74c3c', fontweight='bold',
            )
            ax.axis('off')
        plt.tight_layout()
        _save(fig, 'error_analysis.png')
        plt.close(fig)

    probs = [r['prob_dirty'] for r in results]
    truths = [r['true'] for r in results]
    correct = [r['correct'] for r in results]

    fig, ax = plt.subplots(figsize=(10, 4))
    for i, (p, c, t) in enumerate(zip(probs, correct, truths)):
        col = '#2ecc71' if c else '#e74c3c'
        mk = 'o' if t == 1 else 's'
        ax.scatter(i, p, color=col, marker=mk, s=100, zorder=3,
                   edgecolors='black', linewidth=0.5)
    ax.axhline(0.5, color='gray', linestyle='--', label='Порог 0.5')
    ax.set_xlabel('Индекс изображения')
    ax.set_ylabel('P(dirty)')
    ax.set_title('Уверенность модели на валидации')
    ax.legend(handles=[
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ecc71',
               markersize=10, label='Грязный (верно)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#2ecc71',
               markersize=10, label='Чистый (верно)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c',
               markersize=10, label='Ошибка'),
    ])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, 'prediction_confidence.png')
    plt.close(fig)


def plot_test_predictions(results_df, test_probs, test_preds):
    fig, axes = plt.subplots(2, 8, figsize=(22, 6))
    fig.suptitle('Примеры предсказаний на тесте', fontsize=13, fontweight='bold')

    for row, cls_idx in enumerate([0, 1]):
        cls_df = results_df[results_df.label == cls_idx].head(8)
        for col, (_, row_data) in enumerate(cls_df.iterrows()):
            ax = axes[row, col]
            ax.imshow(Image.open(TEST_DIR / row_data['filename']))
            conf = row_data['prob_dirty'] if cls_idx == 1 else row_data['prob_clean']
            ax.set_title(f'{CLASS_RU[CLASSES[cls_idx]]}\n{conf:.2f}', fontsize=8)
            ax.axis('off')

    plt.tight_layout()
    _save(fig, 'test_predictions_vis.png')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(test_probs[test_preds == 0], bins=30, alpha=0.7, color='#2ecc71',
            label='Чистые', edgecolor='black', linewidth=0.3)
    ax.hist(test_probs[test_preds == 1], bins=30, alpha=0.7, color='#e74c3c',
            label='Грязные', edgecolor='black', linewidth=0.3)
    ax.axvline(0.5, color='black', linestyle='--', label='Порог 0.5')
    ax.set_xlabel('P(dirty)')
    ax.set_ylabel('Количество')
    ax.set_title('Распределение уверенности на тесте')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, 'test_confidence_dist.png')
    plt.close(fig)
