"""
dataset.py
----------
Lädt das Röntgenbild-Dataset für die Frakturerkennung.

Erwartete Ordnerstruktur (Standard Kaggle-Format):
    data/
    ├── train/
    │   ├── fractured/
    │   └── not fractured/
    └── val/
        ├── fractured/
        └── not fractured/

Alternativ (automatischer Split):
    data/
    ├── fractured/
    └── not fractured/
"""

import os
from pathlib import Path

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Beschädigte Bilder tolerieren

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),   # X-Ray oft graustufen → RGB
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

VAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ---------------------------------------------------------------------------
# Loader factory
# ---------------------------------------------------------------------------

def get_dataloaders(data_dir: str, batch_size: int = 32, val_split: float = 0.2,
                    num_workers: int = 2):
    """
    Erstellt Train- und Validierungs-DataLoader.

    Unterstützt zwei Ordnerstrukturen:
      1. data_dir/train/ und data_dir/val/  (Kaggle-Standard)
      2. data_dir/fractured/ und data_dir/not fractured/  (ein einzelner Ordner)

    Gibt zurück:
        train_loader, val_loader, class_names
    """
    data_path = Path(data_dir)

    # -- Struktur erkennen ---------------------------------------------------
    train_path = data_path / "train"
    val_path   = data_path / "val"

    if train_path.exists() and val_path.exists():
        # Struktur 1: vorgegebener Split
        train_dataset = datasets.ImageFolder(train_path, transform=TRAIN_TRANSFORMS)
        val_dataset   = datasets.ImageFolder(val_path,   transform=VAL_TRANSFORMS)
    else:
        # Struktur 2: manueller Split
        full_dataset = datasets.ImageFolder(data_path, transform=TRAIN_TRANSFORMS)
        n_val   = int(len(full_dataset) * val_split)
        n_train = len(full_dataset) - n_val
        train_raw, val_raw = random_split(full_dataset, [n_train, n_val])

        # Val bekommt andere Transforms
        val_raw.dataset = datasets.ImageFolder(data_path, transform=VAL_TRANSFORMS)
        train_dataset, val_dataset = train_raw, val_raw

    class_names = (train_dataset.classes
                   if hasattr(train_dataset, "classes")
                   else train_dataset.dataset.classes)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers, pin_memory=True)

    print(f"[Dataset]  Train: {len(train_dataset):,}  |  Val: {len(val_dataset):,}")
    print(f"[Dataset]  Klassen: {class_names}")
    return train_loader, val_loader, class_names