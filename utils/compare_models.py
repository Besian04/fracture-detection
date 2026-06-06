"""
compare_models.py
-----------------
Vergleicht CNN-Baseline vs. ResNet50 Transfer Learning
anhand aller Metriken und erstellt eine Vergleichsvisualisierung.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             accuracy_score, roc_auc_score)


def compute_metrics(labels, preds, probs):
    return {
        "Accuracy":  accuracy_score(labels, preds),
        "Precision": precision_score(labels, preds, zero_division=0),
        "Recall":    recall_score(labels, preds, zero_division=0),
        "F1-Score":  f1_score(labels, preds, zero_division=0),
        "AUC-ROC":   roc_auc_score(labels, probs),
    }


def compare_and_plot(results: dict, save_path: str = None):
    """
    Args:
        results: {
            "CNN":     {"labels": ..., "preds": ..., "probs": ...},
            "ResNet50":{"labels": ..., "preds": ..., "probs": ...},
        }
    """
    metrics_per_model = {}
    for name, data in results.items():
        metrics_per_model[name] = compute_metrics(
            data["labels"], data["preds"], data["probs"]
        )

    metric_names = list(next(iter(metrics_per_model.values())).keys())
    model_names  = list(metrics_per_model.keys())

    # Konsolen-Ausgabe
    print("\n" + "="*60)
    print("  MODELLVERGLEICH")
    print("="*60)
    header = f"{'Metrik':<14}" + "".join(f"{m:>14}" for m in model_names)
    print(header)
    print("-"*60)
    for metric in metric_names:
        row = f"{metric:<14}"
        for model in model_names:
            row += f"{metrics_per_model[model][metric]:>14.4f}"
        print(row)
    print("="*60)

    # Balkendiagramm
    x    = np.arange(len(metric_names))
    w    = 0.35
    colors = ["#4C72B0", "#DD8452"]

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (model_name, color) in enumerate(zip(model_names, colors)):
        vals = [metrics_per_model[model_name][m] for m in metric_names]
        bars = ax.bar(x + i * w, vals, width=w, label=model_name,
                      color=color, edgecolor="white", linewidth=0.8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8.5)

    ax.set_xticks(x + w / 2)
    ax.set_xticklabels(metric_names, fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Modellvergleich: CNN vs. ResNet50 Transfer Learning",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"\n  Vergleichsdiagramm gespeichert: {save_path}")
    plt.show()
    return metrics_per_model
