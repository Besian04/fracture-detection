"""
trainer.py
----------
Trainingsschleife mit Early Stopping, Learning-Rate-Scheduling
und vollständiger Metrik-Protokollierung.
"""

import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau


class EarlyStopping:
    """Stoppt Training, wenn Validierungsloss sich nicht verbessert."""

    def __init__(self, patience: int = 5, min_delta: float = 1e-4,
                 checkpoint_path: str = "best_model.pth"):
        self.patience   = patience
        self.min_delta  = min_delta
        self.path       = checkpoint_path
        self.best_loss  = float("inf")
        self.counter    = 0
        self.best_epoch = 0

    def __call__(self, val_loss: float, model: nn.Module, epoch: int) -> bool:
        """Gibt True zurück, wenn Training gestoppt werden soll."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss  = val_loss
            self.counter    = 0
            self.best_epoch = epoch
            torch.save(model.state_dict(), self.path)
            print(f"  ✓ Bestes Modell gespeichert (Val-Loss: {val_loss:.4f})")
        else:
            self.counter += 1
            print(f"  EarlyStopping: {self.counter}/{self.patience} (best: {self.best_loss:.4f})")
            if self.counter >= self.patience:
                print(f"\n  → Training gestoppt. Bestes Modell aus Epoch {self.best_epoch}.")
                return True
        return False


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds      = (torch.sigmoid(outputs) >= 0.5).float()
        correct    += (preds == labels).sum().item()
        total      += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        outputs    = model(images)
        loss       = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)

        preds   = (torch.sigmoid(outputs) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total   += images.size(0)

    return total_loss / total, correct / total


def train(model, train_loader, val_loader, config: dict, results_dir: str = "results"):
    """
    Haupttrainingsfunktion.

    Args:
        model:        PyTorch-Modell
        train_loader: DataLoader (Training)
        val_loader:   DataLoader (Validierung)
        config:       dict mit Hyperparametern
        results_dir:  Ordner für Checkpoints

    Returns:
        history: dict mit Listen für loss/acc (train + val)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Trainer]  Device: {device}")
    model.to(device)

    Path(results_dir).mkdir(parents=True, exist_ok=True)
    model_name    = config.get("model_name", "model")
    checkpoint    = f"{results_dir}/{model_name}_best.pth"

    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.get("lr", 1e-3),
        weight_decay=config.get("weight_decay", 1e-4),
    )
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([config.get("pos_weight", 1.0)]).to(device)
    )
    scheduler     = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    early_stop    = EarlyStopping(patience=config.get("patience", 7),
                                  checkpoint_path=checkpoint)

    history = {"train_loss": [], "val_loss": [],
               "train_acc":  [], "val_acc":  []}

    epochs = config.get("epochs", 30)
    print(f"[Trainer]  Start Training: {epochs} Epochen\n")

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - t0
        print(f"Epoch {epoch:02d}/{epochs}  "
              f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}  |  "
              f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}  "
              f"({elapsed:.1f}s)")

        if early_stop(val_loss, model, epoch):
            break

    # Bestes Modell laden
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    print(f"\n[Trainer]  Bestes Modell geladen: {checkpoint}")
    return history, device
