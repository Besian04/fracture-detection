import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import torch

from utils.dataset        import get_dataloaders
from utils.trainer        import train
from utils.evaluation     import full_evaluation
from utils.gradcam        import batch_gradcam
from utils.compare_models import compare_and_plot
from models.cnn_model     import FractureCNN
from models.transfer_model import FractureResNet

DEFAULT_CONFIG = {
    "epochs": 30, "batch_size": 32, "lr": 1e-3, "weight_decay": 1e-4,
    "patience": 7, "pos_weight": 1.5, "resnet_lr": 3e-4,
    "unfreeze_blocks": 2, "val_split": 0.2, "num_workers": 2,
}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",    type=str, default="./data")
    p.add_argument("--results_dir", type=str, default="./results")
    p.add_argument("--epochs",      type=int, default=DEFAULT_CONFIG["epochs"])
    p.add_argument("--batch_size",  type=int, default=DEFAULT_CONFIG["batch_size"])
    p.add_argument("--skip_cnn",    action="store_true")
    p.add_argument("--skip_resnet", action="store_true")
    p.add_argument("--gradcam_samples", type=int, default=6)
    return p.parse_args()

def main():
    args = parse_args()
    cfg  = DEFAULT_CONFIG.copy()
    cfg["epochs"]     = args.epochs
    cfg["batch_size"] = args.batch_size

    print("\n🩻  Knochenfraktur-Erkennung – Deep Learning\n")

    train_loader, val_loader, class_names = get_dataloaders(
        data_dir=args.data_dir, batch_size=cfg["batch_size"],
        val_split=cfg["val_split"], num_workers=cfg["num_workers"],
    )
    print(f"[Info] Klassen: {dict(enumerate(class_names))}")

    comparison_results = {}

    if not args.skip_cnn:
        print("\n--- MODELL 1: Custom CNN ---")
        cnn_model = FractureCNN(dropout=0.5)
        cnn_cfg = {**cfg, "model_name": "CNN", "lr": cfg["lr"]}
        cnn_history, device = train(cnn_model, train_loader, val_loader, cnn_cfg, results_dir=args.results_dir)
        cnn_labels, cnn_preds, cnn_probs, _ = full_evaluation(
            cnn_model, val_loader, device, class_names, "CNN", cnn_history, args.results_dir)
        target_layer_cnn = list(cnn_model.features.children())[-1].block[0]
        batch_gradcam(cnn_model, target_layer_cnn, val_loader, device,
                      n_samples=args.gradcam_samples, class_names=class_names,
                      save_path=f"{args.results_dir}/CNN_gradcam.png")
        comparison_results["CNN"] = {"labels": cnn_labels, "preds": cnn_preds, "probs": cnn_probs}

    if not args.skip_resnet:
        print("\n--- MODELL 2: ResNet50 Transfer Learning ---")
        resnet_model = FractureResNet(dropout=0.4, unfreeze_blocks=cfg["unfreeze_blocks"])
        resnet_model.get_trainable_params()
        resnet_cfg = {**cfg, "model_name": "ResNet50", "lr": cfg["resnet_lr"]}
        resnet_history, device = train(resnet_model, train_loader, val_loader, resnet_cfg, results_dir=args.results_dir)
        resnet_labels, resnet_preds, resnet_probs, _ = full_evaluation(
            resnet_model, val_loader, device, class_names, "ResNet50", resnet_history, args.results_dir)
        batch_gradcam(resnet_model, resnet_model.last_conv, val_loader, device,
                      n_samples=args.gradcam_samples, class_names=class_names,
                      save_path=f"{args.results_dir}/ResNet50_gradcam.png")
        comparison_results["ResNet50"] = {"labels": resnet_labels, "preds": resnet_preds, "probs": resnet_probs}

    if len(comparison_results) >= 2:
        compare_and_plot(comparison_results, save_path=f"{args.results_dir}/model_comparison.png")

    print("\n✅ Pipeline abgeschlossen. Ergebnisse in:", args.results_dir)

if __name__ == "__main__":
    main()
