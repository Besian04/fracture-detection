"""
evaluation.py
-------------
Vollständige Modell-Evaluation:
    - Accuracy, Precision, Recall, F1-Score
    - Confusion Matrix (Visualisierung)
    - ROC-Kurve
    - Trainings-Kurven (Loss & Accuracy)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve)


@torch.no_grad()
def get_predictions(model, loader, device):
    """Sammelt alle Vorhersagen und Ground-Truth-Labels."""
    model.eval()
    all_probs, all_labels = [], []

    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        probs  = torch.sigmoid(logits).squeeze(1).cpu()
        all_probs.append(probs)
        all_labels.append(labels)

    probs  = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy()
    preds  = (probs >= 0.5).astype(int)
    return labels, preds, probs


def print_metrics(labels, preds, class_names=("No Fracture", "Fracture")):
    """Gibt Klassifikationsbericht auf der Konsole aus."""
    print("\n" + "="*55)
    print("  KLASSIFIKATIONSBERICHT")
    print("="*55)
    print(classification_report(labels, preds, target_names=class_names, digits=4))


def plot_confusion_matrix(labels, preds, class_names, save_path: str = None):
    """Zeichnet und speichert die Confusion Matrix."""
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Confusion Matrix gespeichert: {save_path}")
    plt.show()
    return cm


def plot_roc_curve(labels, probs, model_name: str = "", save_path: str = None):
    """Zeichnet ROC-Kurve und berechnet AUC."""
    fpr, tpr, _ = roc_curve(labels, probs)
    auc         = roc_auc_score(labels, probs)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC-Kurve  {model_name}", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  ROC-Kurve gespeichert: {save_path}")
    plt.show()
    return auc


def plot_training_curves(history: dict, model_name: str = "", save_path: str = None):
    """Zeichnet Loss- und Accuracy-Kurven."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    axes[0].plot(epochs, history["train_loss"], label="Train Loss", marker="o", ms=4)
    axes[0].plot(epochs, history["val_loss"],   label="Val Loss",   marker="s", ms=4)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss (BCE)")
    axes[0].set_title(f"Loss  –  {model_name}")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history["train_acc"], label="Train Acc", marker="o", ms=4)
    axes[1].plot(epochs, history["val_acc"],   label="Val Acc",   marker="s", ms=4)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title(f"Accuracy  –  {model_name}")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Trainings-Kurven gespeichert: {save_path}")
    plt.show()


def full_evaluation(model, val_loader, device, class_names, model_name: str,
                    history: dict, results_dir: str = "results"):
    """
    Komplette Evaluation eines Modells:
    Metriken + Confusion Matrix + ROC + Trainings-Kurven.
    """
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    labels, preds, probs = get_predictions(model, val_loader, device)

    print_metrics(labels, preds, class_names)

    plot_confusion_matrix(
        labels, preds, class_names,
        save_path=f"{results_dir}/{model_name}_confusion_matrix.png"
    )

    auc = plot_roc_curve(
        labels, probs, model_name=model_name,
        save_path=f"{results_dir}/{model_name}_roc_curve.png"
    )
    print(f"  AUC-ROC: {auc:.4f}")

    plot_training_curves(
        history, model_name=model_name,
        save_path=f"{results_dir}/{model_name}_training_curves.png"
    )

    return labels, preds, probs, auc
