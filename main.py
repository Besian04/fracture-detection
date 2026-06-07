# main.py
# -------
# Dies ist unser Hauptskript zur Ausführung der gesamten Pipeline:
# Laden des Datensatzes -> Training beider Modelle -> Auswertung der Metriken -> Grad-CAM visualisieren -> Modelle vergleichen.

import argparse
import sys
import os

# Damit Python das aktuelle Skript-Verzeichnis im Suchpfad hat
sys.path.insert(0, os.path.dirname(__file__))

import torch

# Wir importieren unsere Hilfsmodule für Dataset, Training und Evaluation
from utils.dataset        import get_dataloaders
from utils.trainer        import train
from utils.evaluation     import full_evaluation
from utils.gradcam        import batch_gradcam
from utils.compare_models import compare_and_plot
from models.cnn_model     import FractureCNN
from models.transfer_model import FractureResNet

# Unsere Standard-Konfiguration mit allen wichtigen Hyperparametern für beide Modelle
DEFAULT_CONFIG = {
    "epochs": 30,             # Maximale Epochenzahl für das Training
    "batch_size": 32,         # Anzahl der Bilder pro Batch
    "lr": 1e-3,               # Lernrate für das Custom-CNN
    "weight_decay": 1e-4,     # L2-Regularisierung, um Overfitting zu reduzieren
    "patience": 7,            # patience für Early Stopping (warten auf Verbesserung des Val-Loss)
    "pos_weight": 1.5,        # Gewichtung für positive Klasse (Bruch) im Loss, da wir Recall priorisieren
    "resnet_lr": 3e-4,        # Kleinere Lernrate für ResNet50 Transfer-Learning
    "unfreeze_blocks": 2,     # Anzahl der letzten ResNet-Blöcke, die beim Feintuning trainiert werden
    "val_split": 0.2,         # 20% der Daten für die Validierung nutzen
    "num_workers": 2,         # Multiprocessing für schnelleres Laden der Bilder
}

def parse_args():
    # Ermöglicht das Anpassen von Pfaden und Epochen direkt beim Aufruf über das Terminal
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",    type=str, default="./data") # Wo liegen die Röntgenbilder?
    p.add_argument("--results_dir", type=str, default="./results") # Wo sollen Modelle/Plots gespeichert werden?
    p.add_argument("--epochs",      type=int, default=DEFAULT_CONFIG["epochs"])
    p.add_argument("--batch_size",  type=int, default=DEFAULT_CONFIG["batch_size"])
    p.add_argument("--skip_cnn",    action="store_true") # Custom CNN überspringen?
    p.add_argument("--skip_resnet", action="store_true") # ResNet50 überspringen?
    p.add_argument("--gradcam_samples", type=int, default=6) # Anzahl der Röntgenbilder für Grad-CAM-Prüfung
    return p.parse_args()

def main():
    args = parse_args()
    cfg  = DEFAULT_CONFIG.copy()
    cfg["epochs"]     = args.epochs
    cfg["batch_size"] = args.batch_size

    print("\nKnochenfraktur-Erkennung - Deep Learning\n")

    # 1. Schritt: Dataloader erzeugen (Trainings- und Validierungsdaten laden und vorbereiten)
    train_loader, val_loader, class_names = get_dataloaders(
        data_dir=args.data_dir, batch_size=cfg["batch_size"],
        val_split=cfg["val_split"], num_workers=cfg["num_workers"],
    )
    print(f"[Info] Klassen: {dict(enumerate(class_names))}")

    comparison_results = {}

    # 2. Schritt: Modell 1 (Custom CNN) trainieren und evaluieren
    if not args.skip_cnn:
        print("\n--- MODELL 1: Custom CNN ---")
        # Custom-CNN-Instanz mit 50% Dropout initialisieren
        cnn_model = FractureCNN(dropout=0.5)
        # Hyperparameter konfigurieren
        cnn_cfg = {**cfg, "model_name": "CNN", "lr": cfg["lr"]}
        # Trainingsschleife starten
        cnn_history, device = train(cnn_model, train_loader, val_loader, cnn_cfg, results_dir=args.results_dir)
        # Vollständige Metriken auf dem Validierungsset berechnen und Plots erstellen
        cnn_labels, cnn_preds, cnn_probs, _ = full_evaluation(
            cnn_model, val_loader, device, class_names, "CNN", cnn_history, args.results_dir)
        # Für Grad-CAM greifen wir die letzte Convolutional-Schicht des Custom-CNNs ab
        target_layer_cnn = list(cnn_model.features.children())[-1].block[0]
        # Grad-CAM Heatmaps für eine Stichprobe von Bildern generieren und speichern
        batch_gradcam(cnn_model, target_layer_cnn, val_loader, device,
                      n_samples=args.gradcam_samples, class_names=class_names,
                      save_path=f"{args.results_dir}/CNN_gradcam.png")
        # Ergebnisse für den späteren Vergleich sichern
        comparison_results["CNN"] = {"labels": cnn_labels, "preds": cnn_preds, "probs": cnn_probs}

    # 3. Schritt: Modell 2 (ResNet50 Transfer Learning) trainieren und evaluieren
    if not args.skip_resnet:
        print("\n--- MODELL 2: ResNet50 Transfer Learning ---")
        # ResNet50 initialisieren, letzte 2 Blöcke + Klassifikationskopf werden trainiert
        resnet_model = FractureResNet(dropout=0.4, unfreeze_blocks=cfg["unfreeze_blocks"])
        # Info über trainierbare Parameter ausgeben
        resnet_model.get_trainable_params()
        # Konfiguration für ResNet50
        resnet_cfg = {**cfg, "model_name": "ResNet50", "lr": cfg["resnet_lr"]}
        # Training mit kleinerer Lernrate ausführen
        resnet_history, device = train(resnet_model, train_loader, val_loader, resnet_cfg, results_dir=args.results_dir)
        # Metriken berechnen und speichern
        resnet_labels, resnet_preds, resnet_probs, _ = full_evaluation(
            resnet_model, val_loader, device, class_names, "ResNet50", resnet_history, args.results_dir)
        # Grad-CAM Heatmaps für ResNet50 erzeugen
        batch_gradcam(resnet_model, resnet_model.last_conv, val_loader, device,
                      n_samples=args.gradcam_samples, class_names=class_names,
                      save_path=f"{args.results_dir}/ResNet50_gradcam.png")
        # Ergebnisse speichern
        comparison_results["ResNet50"] = {"labels": resnet_labels, "preds": resnet_preds, "probs": resnet_probs}

    # 4. Schritt: Wenn beide Modelle trainiert wurden, erstellen wir den direkten Vergleichs-Plot
    if len(comparison_results) >= 2:
        compare_and_plot(comparison_results, save_path=f"{args.results_dir}/model_comparison.png")

    print("\nPipeline abgeschlossen. Ergebnisse in:", args.results_dir)

if __name__ == "__main__":
    main()
