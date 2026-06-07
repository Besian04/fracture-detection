# dataset.py
# ----------
# In dieser Datei strukturieren und laden wir unsere Röntgenbilder.
# Wir transformieren die Bilddaten so, dass sie für die neuronalen Netze optimal lesbar sind
# und führen Datenaugmentation durch, um Overfitting zu reduzieren.

import os
from pathlib import Path

from PIL import ImageFile
# Einige Röntgenbilder im Dataset könnten unvollständig übertragen worden sein.
# Mit dieser Einstellung toleriert PIL beschädigte Bilddateien und bricht das Training nicht ab.
ImageFile.LOAD_TRUNCATED_IMAGES = True

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


# ---------------------------------------------------------------------------
# Bildtransformationen (Transforms) und Datenaugmentation
# ---------------------------------------------------------------------------

# Datenerweiterung (Augmentation) für die Trainingsdaten:
# Wir verändern die Bilder leicht bei jedem Laden, damit das Modell robustere Merkmale lernt
# und sich nicht auf bestimmte Ausrichtungen oder Lichtverhältnisse verlässt.
TRAIN_TRANSFORMS = transforms.Compose([
    # Bildgröße auf 224x224 anpassen (Standard für ResNet)
    transforms.Resize((224, 224)),
    # Röntgenbilder sind oft Graustufen. Wir konvertieren sie hier in 3 identische RGB-Kanäle,
    # da ResNet50 zwingend 3 Eingabekanäle erwartet.
    transforms.Grayscale(num_output_channels=3),
    # Zufälliges horizontales Spiegeln mit 50% Wahrscheinlichkeit
    transforms.RandomHorizontalFlip(),
    # Zufällige Rotation um bis zu 10 Grad, um ungenaue Röntgenaufnahmen zu simulieren
    transforms.RandomRotation(10),
    # Zufällige Helligkeits- und Kontrastschwankungen von bis zu 20%
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    # Bild in einen PyTorch-Tensor konvertieren (skaliert Pixelwerte auf [0.0, 1.0])
    transforms.ToTensor(),
    # Normalisierung mit Standardwerten (ImageNet-Mittelwert und Standardabweichung)
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Validierungsdaten-Transformationen:
# Hier machen wir KEINE Augmentation (keine Spiegelung, Drehung etc.), da wir die echte Leistung
# auf unveränderten Daten messen wollen. Wir passen nur die Größe an und normalisieren.
VAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ---------------------------------------------------------------------------
# DataLoader-Fabrik
# ---------------------------------------------------------------------------

def get_dataloaders(data_dir: str, batch_size: int = 32, val_split: float = 0.2,
                    num_workers: int = 2):
    """
    Diese Funktion lädt die Bildordner und verpackt sie in PyTorch DataLoader.
    
    Wir unterstützen flexibel zwei Dataset-Strukturen:
      1. Bereits getrennt: 'data/train/' und 'data/val/' (Standard z.B. bei Kaggle)
      2. Ein einzelner Ordner: 'data/fractured/' und 'data/not fractured/' -> Hier splitten wir automatisch 80/20.
    """
    data_path = Path(data_dir)

    # Pfade für eventuell existierende train/val Unterordner bauen
    train_path = data_path / "train"
    val_path   = data_path / "val"

    # Prüfen, ob bereits getrennte Ordner für Training und Validierung existieren
    if train_path.exists() and val_path.exists():
        # Struktur 1: Vorgegebener Split wird direkt geladen
        train_dataset = datasets.ImageFolder(train_path, transform=TRAIN_TRANSFORMS)
        val_dataset   = datasets.ImageFolder(val_path,   transform=VAL_TRANSFORMS)
    else:
        # Struktur 2: Wir haben nur einen gemeinsamen Ordner und müssen selbst aufteilen
        full_dataset = datasets.ImageFolder(data_path, transform=TRAIN_TRANSFORMS)
        # Berechnung der Anzahl der Validierungsbilder basierend auf 'val_split' (z.B. 20%)
        n_val   = int(len(full_dataset) * val_split)
        n_train = len(full_dataset) - n_val
        # Zufälliger Split der Indizes
        train_raw, val_raw = random_split(full_dataset, [n_train, n_val])

        # WICHTIG: Das Validierungsset darf keine Trainings-Augmentationen erhalten!
        # Deshalb weisen wir dem Validierungs-Subset nachträglich VAL_TRANSFORMS zu.
        val_raw.dataset = datasets.ImageFolder(data_path, transform=VAL_TRANSFORMS)
        train_dataset, val_dataset = train_raw, val_raw

    # Klassen-Namen extrahieren (wird aus den Ordnernamen 'fractured' / 'not fractured' ermittelt)
    class_names = (train_dataset.classes
                   if hasattr(train_dataset, "classes")
                   else train_dataset.dataset.classes)

    # DataLoader erstellen: Sie laden die Bilder im Hintergrund in Mini-Batches (hier 32 Bilder) auf die CPU/GPU.
    # shuffle=True sorgt dafür, dass die Reihenfolge der Trainingsbilder in jeder Epoche durchgemischt wird.
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers, pin_memory=True)
    # shuffle=False bei Validierung, da die Reihenfolge hier keine Rolle spielt.
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers, pin_memory=True)

    # Statusberichte in der Konsole ausgeben
    print(f"[Dataset]  Train: {len(train_dataset):,}  |  Val: {len(val_dataset):,}")
    print(f"[Dataset]  Klassen: {class_names}")
    return train_loader, val_loader, class_names