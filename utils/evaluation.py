# evaluation.py

# In dieser Datei werten wir die Leistung unserer Netze statistisch und visuell aus.
# Wir berechnen die Standardmetriken (Accuracy, Precision, Recall, F1)
# und zeichnen Plots wie die Confusion Matrix, die ROC-Kurve und die Loss/Acc Lernkurven.

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve)


@torch.no_grad() # Keine Gradienten berechnen, da wir hier nur testen und nicht trainieren
def get_predictions(model, loader, device):
    """
    Diese Funktion sammelt die Vorhersagen (Wahrscheinlichkeiten) des Modells 
    und die tatsächlichen Klassenlabels über den gesamten DataLoader hinweg.
    """
    model.eval() # Modell in Evaluierungsmodus setzen
    all_probs, all_labels = [], []

    for images, labels in loader:
        images = images.to(device)
        # 1. Forward Pass auf der Grafikkarte ausführen
        logits = model(images)
        # 2. Logits mit Sigmoid in Wahrscheinlichkeiten umrechnen (Bereich 0 bis 1)
        probs  = torch.sigmoid(logits).squeeze(1).cpu()
        all_probs.append(probs)
        all_labels.append(labels)

    # Alle Batches zu einem einzigen Numpy-Array zusammenfügen
    probs  = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy()
    # Entscheidungsschwelle bei 0.5: Alles darüber gilt als Bruch (1), darunter als gesund (0)
    preds  = (probs >= 0.5).astype(int)
    return labels, preds, probs


def print_metrics(labels, preds, class_names=("No Fracture", "Fracture")):
    """
    Erzeugt einen tabellarischen Klassifikationsbericht in der Konsole.
    Zeigt Precision, Recall und F1-Score für beide Klassen an.
    """
    print("\n" + "="*55)
    print("  KLASSIFIKATIONSBERICHT")
    print("="*55)
    # Das Modul scikit-learn berechnet die Metriken automatisch
    print(classification_report(labels, preds, target_names=class_names, digits=4))


def plot_confusion_matrix(labels, preds, class_names, save_path: str = None):
    """
    Zeichnet die Confusion Matrix (Wahrheitsmatrix) und speichert sie als PNG ab.
    Sie zeigt, wie viele Frakturen korrekt als solche erkannt wurden (True Positives),
    wie oft das Modell falsch alarmiert hat (False Positives) und was übersehen wurde (False Negatives).
    """
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    # Nutzt scikit-learns vorgefertigten Visualisierer
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    # Auf Festplatte sichern
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Confusion Matrix gespeichert: {save_path}")
    plt.show()
    return cm


def plot_roc_curve(labels, probs, model_name: str = "", save_path: str = None):
    """
    Zeichnet die ROC-Kurve (Receiver Operating Characteristic) und berechnet den AUC-Wert.
    Zeigt die Qualität des Modells über alle möglichen Entscheidungsschwellen hinweg.
    """
    # Falsch-Positiv-Rate und Richtig-Positiv-Rate über Schwellenwerte berechnen
    fpr, tpr, _ = roc_curve(labels, probs)
    # AUC berechnen (Fläche unter der Kurve; 1.0 = perfekt, 0.5 = Raten)
    auc         = roc_auc_score(labels, probs)

    fig, ax = plt.subplots(figsize=(6, 5))
    # Die ROC-Kurve zeichnen
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.4f}")
    # Die diagonale Zufallslinie (Baseline für blindes Raten) einzeichnen
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
    """
    Zeichnet den Verlauf von Loss (Verlust) und Accuracy (Genauigkeit) während des Trainings.
    Das hilft uns, Overfitting (Überanpassung) visuell zu identifizieren.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Linker Plot: Verlauf der Loss-Kurven (Train vs. Validation)
    axes[0].plot(epochs, history["train_loss"], label="Train Loss", marker="o", ms=4)
    axes[0].plot(epochs, history["val_loss"],   label="Val Loss",   marker="s", ms=4)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss (BCE)")
    axes[0].set_title(f"Loss  –  {model_name}")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Rechter Plot: Verlauf der Genauigkeit (Train vs. Validation)
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
    Führt die komplette Evaluation für das Modell aus:
    Berechnet die Metriken, zeichnet und speichert alle Grafiken im 'results'-Verzeichnis.
    """
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    # 1. Vorhersagen sammeln
    labels, preds, probs = get_predictions(model, val_loader, device)

    # 2. Textmetriken ausgeben
    print_metrics(labels, preds, class_names)

    # 3. Confusion Matrix zeichnen und sichern
    plot_confusion_matrix(
        labels, preds, class_names,
        save_path=f"{results_dir}/{model_name}_confusion_matrix.png"
    )

    # 4. ROC-Kurve zeichnen und sichern
    auc = plot_roc_curve(
        labels, probs, model_name=model_name,
        save_path=f"{results_dir}/{model_name}_roc_curve.png"
    )
    print(f"  AUC-ROC: {auc:.4f}")

    # 5. Lernkurven zeichnen und sichern
    plot_training_curves(
        history, model_name=model_name,
        save_path=f"{results_dir}/{model_name}_training_curves.png"
    )

    return labels, preds, probs, auc
