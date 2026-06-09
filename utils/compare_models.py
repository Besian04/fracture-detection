# compare_models.py

# In dieser Datei vergleichen wir unsere beiden trainierten Modelle
# (das selbstgebaute CNN und das ResNet50 Transfer-Learning-Modell)
# direkt anhand aller Leistungswerte und erzeugen ein vergleichendes Balkendiagramm.

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             accuracy_score, roc_auc_score)


def compute_metrics(labels, preds, probs):
    """
    Hilfsfunktion zur Berechnung aller wichtigen Klassifikationsmetriken.
    Verwendet die Standard-Mathematikfunktionen aus scikit-learn.
    """
    return {
        "Accuracy":  accuracy_score(labels, preds),                   # Gesamte Genauigkeit
        "Precision": precision_score(labels, preds, zero_division=0),  # Anteil echter Brüche unter allen Bruch-Vorhersagen
        "Recall":    recall_score(labels, preds, zero_division=0),     # Anteil gefundener Brüche an allen echten Brüchen
        "F1-Score":  f1_score(labels, preds, zero_division=0),         # Harmonisiertes Mittel aus Precision und Recall
        "AUC-ROC":   roc_auc_score(labels, probs),                     # Fläche unter der ROC-Kurve (Modellstärke allgemein)
    }


def compare_and_plot(results: dict, save_path: str = None):
    """
    Berechnet die Metriken für beide Modelle, gibt sie tabellarisch in der Konsole aus
    und erzeugt ein schickes Balkendiagramm, das wir im Modell-Dashboard einbinden.
    
    Args:
        results: Ein Dictionary der Form:
            {
                "CNN":     {"labels": ..., "preds": ..., "probs": ...},
                "ResNet50":{"labels": ..., "preds": ..., "probs": ...},
            }
    """
    metrics_per_model = {}
    # Metriken für jedes Modell berechnen
    for name, data in results.items():
        metrics_per_model[name] = compute_metrics(
            data["labels"], data["preds"], data["probs"]
        )

    # Bezeichnungen holen
    metric_names = list(next(iter(metrics_per_model.values())).keys())
    model_names  = list(metrics_per_model.keys())

    # --- 1. Konsolen-Ausgabe ---
    # Erzeugt eine formatiert ausgerichtete Tabelle im Terminal
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

    # --- 2. Balkendiagramm erzeugen ---
    x    = np.arange(len(metric_names))  # X-Positionen für die Metriken
    w    = 0.35                         # Breite der Balken
    colors = ["#4C72B0", "#DD8452"]     # Blau für CNN, Orange für ResNet50

    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Balken für jedes Modell zeichnen
    for i, (model_name, color) in enumerate(zip(model_names, colors)):
        vals = [metrics_per_model[model_name][m] for m in metric_names]
        # Balkengruppe zeichnen (leicht versetzt per i * w)
        bars = ax.bar(x + i * w, vals, width=w, label=model_name,
                      color=color, edgecolor="white", linewidth=0.8)
        # Genauen Zahlenwert über jeden Balken schreiben
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8.5)

    # Achsen-Beschriftungen und Layout
    ax.set_xticks(x + w / 2)
    ax.set_xticklabels(metric_names, fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Modellvergleich: CNN vs. ResNet50 Transfer Learning",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    # Auf Wunsch das fertige Diagramm sichern
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"\n  Vergleichsdiagramm gespeichert: {save_path}")
    plt.show()
    return metrics_per_model
