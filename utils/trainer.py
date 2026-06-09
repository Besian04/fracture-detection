# trainer.py

# In dieser Datei steuert die eigentliche Trainingsschleife das Training der Netze.
# Außerdem implementieren wir hier ein "Early Stopping", um Overfitting abzufangen,
# und steuern das dynamische Senken der Lernrate (LR Scheduling).

import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau


class EarlyStopping:
    """
    Early Stopping überwacht den Validierungsverlust (Val-Loss).
    Wenn sich dieser über eine bestimmte Epochenzahl (patience) hinweg nicht verbessert,
    bricht das Training automatisch ab, um ein Überlernen (Overfitting) zu verhindern.
    """

    def __init__(self, patience: int = 5, min_delta: float = 1e-4,
                 checkpoint_path: str = "best_model.pth"):
        self.patience   = patience            # Wie viele Epochen warten wir ohne Verbesserung?
        self.min_delta  = min_delta           # Mindeständerung, um als echte Verbesserung zu gelten
        self.path       = checkpoint_path     # Speicherpfad für das beste Modell
        self.best_loss  = float("inf")        # Initialer Bestwert des Verlusts (unendlich hoch)
        self.counter    = 0                   # Zähler für Epochen ohne Verbesserung
        self.best_epoch = 0                   # Speichert die Epoche, aus der das beste Modell stammt

    def __call__(self, val_loss: float, model: nn.Module, epoch: int) -> bool:
        """Gibt True zurück, wenn das Training gestoppt werden soll (Stoppsignal)."""
        # Wenn der Verlust sinkt (besser wird)
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss  = val_loss
            self.counter    = 0
            self.best_epoch = epoch
            # Das beste Modell sofort auf der Festplatte sichern
            torch.save(model.state_dict(), self.path)
            print(f"  [OK] Bestes Modell gespeichert (Val-Loss: {val_loss:.4f})")
        else:
            # Keine Verbesserung -> Zähler erhöhen
            self.counter += 1
            print(f"  EarlyStopping: {self.counter}/{self.patience} (best: {self.best_loss:.4f})")
            # Limit erreicht -> Stoppsignal auslösen
            if self.counter >= self.patience:
                print(f"\n  Training gestoppt. Bestes Modell aus Epoch {self.best_epoch}.")
                return True
        return False


def train_one_epoch(model, loader, optimizer, criterion, device):
    """Führt eine einzelne Trainingsepoche über alle Trainingsdaten aus."""
    model.train() # Modell in den Trainingsmodus versetzen (aktiviert Dropout/BatchNorm)
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        # Daten auf das Rechengerät (GPU/CPU) laden
        images = images.to(device)
        # Ziel-Labels anpassen (Binärklassifikation erwartet Dimension [Batchgröße, 1])
        labels = labels.float().unsqueeze(1).to(device)

        # 1. Gradienten zurücksetzen, damit sie sich nicht akkumulieren
        optimizer.zero_grad()
        # 2. Forward Pass: Vorhersagen berechnen
        outputs = model(images)
        # 3. Loss (Verlust) berechnen
        loss    = criterion(outputs, labels)
        # 4. Backward Pass: Gradienten durch Backpropagation berechnen
        loss.backward()
        # 5. Optimierungsschritt: Gewichte des Modells anpassen
        optimizer.step()

        # Metriken für diesen Batch protokollieren
        total_loss += loss.item() * images.size(0)
        # Entscheidungsschwelle bei 50% für die Accuracy-Berechnung
        preds      = (torch.sigmoid(outputs) >= 0.5).float()
        correct    += (preds == labels).sum().item()
        total      += images.size(0)

    # Durchschnittlichen Verlust und Genauigkeit für die Epoche zurückgeben
    return total_loss / total, correct / total


@torch.no_grad() # Deaktiviert die Gradientenberechnung, spart Speicher und Rechenzeit
def evaluate(model, loader, criterion, device):
    """Evaluiert das Modell auf den Validierungsdaten."""
    model.eval() # Modell in den Evaluierungsmodus setzen (deaktiviert Dropout/BatchNorm)
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        # Vorhersagen berechnen (ohne Gewichtsupdate)
        outputs    = model(images)
        loss       = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)

        # Genauigkeit ermitteln
        preds   = (torch.sigmoid(outputs) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total   += images.size(0)

    return total_loss / total, correct / total


def train(model, train_loader, val_loader, config: dict, results_dir: str = "results"):
    """
    Hauptfunktion für das Training über mehrere Epochen hinweg.
    """
    # CUDA-Grafikkarte nutzen falls vorhanden, andernfalls CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Trainer]  Device: {device}")
    model.to(device)

    # Ergebnisse-Verzeichnis erstellen falls nötig
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    model_name    = config.get("model_name", "model")
    checkpoint    = f"{results_dir}/{model_name}_best.pth"

    # Adam-Optimizer konfigurieren. Er passt die Gewichte an.
    # filter() stellt sicher, dass wir nur für freigegebene (trainierbare) Parameter lernen!
    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.get("lr", 1e-3),
        weight_decay=config.get("weight_decay", 1e-4),
    )
    
    # BCEWithLogitsLoss zur binären Klassifikation.
    # pos_weight=1.5 gewichtet Fehler bei Frakturen stärker, um den Recall zu optimieren!
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([config.get("pos_weight", 1.0)]).to(device)
    )
    
    # Scheduler: Reduziert die Lernrate um die Hälfte, wenn der Val-Loss für 3 Epochen stagniert.
    # Dadurch lernt das Modell feiner, je näher es dem Optimum kommt.
    scheduler     = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    
    # Early Stopping instanziieren
    early_stop    = EarlyStopping(patience=config.get("patience", 7),
                                  checkpoint_path=checkpoint)

    # Verlaufsprotokoll (History) initialisieren
    history = {"train_loss": [], "val_loss": [],
               "train_acc":  [], "val_acc":  []}

    epochs = config.get("epochs", 30)
    print(f"[Trainer]  Start Training: {epochs} Epochen\n")

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # 1. Epoche trainieren
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        # 2. Auf Validierungsset prüfen
        val_loss,   val_acc   = evaluate(model, val_loader, criterion, device)
        # 3. Lernrate anpassen basierend auf Validierungsverlust
        scheduler.step(val_loss)

        # Werte im Protokoll speichern
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - t0
        # Fortschritt in der Konsole ausgeben
        print(f"Epoch {epoch:02d}/{epochs}  "
              f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}  |  "
              f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}  "
              f"({elapsed:.1f}s)")

        # Early Stopping überprüfen. Wenn das Stoppsignal kommt, brechen wir die Schleife ab.
        if early_stop(val_loss, model, epoch):
            break

    # Am Ende des Trainings laden wir die Gewichte des besten gespeicherten Modells (nicht der letzten Epoche!)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    print(f"\n[Trainer]  Bestes Modell geladen: {checkpoint}")
    return history, device
