"""
transfer_model.py
-----------------
Transfer-Learning-Modell (Modell 2) auf Basis von ResNet50.

Strategie:
    1. Vortrainierte ResNet50-Gewichte laden (ImageNet)
    2. Alle Layer einfrieren
    3. Letzte FC-Schicht ersetzen (→ binäre Ausgabe)
    4. Optional: Feintuning der letzten ResNet-Blöcke
"""

import torch
import torch.nn as nn
from torchvision import models


class FractureResNet(nn.Module):
    """
    ResNet50 mit angepasstem Klassifikationskopf.

    Input:  (B, 3, 224, 224)
    Output: (B, 1)  → Logits für BCEWithLogitsLoss
    """

    def __init__(self, dropout: float = 0.4, unfreeze_blocks: int = 2):
        """
        Args:
            dropout:        Dropout-Rate im Klassifikationskopf.
            unfreeze_blocks: Anzahl der letzten ResNet-Layer-Gruppen,
                             die fine-getuned werden (0 = nur Head, max 4).
        """
        super().__init__()

        # Vortrainiertes Basismodell laden
        # Versucht ImageNet-Gewichte zu laden; fällt auf zufällige Initialisierung zurück
        try:
            weights = models.ResNet50_Weights.IMAGENET1K_V2
            backbone = models.resnet50(weights=weights)
            print("[ResNet50]  Vortrainierte ImageNet-Gewichte geladen ✓")
        except Exception:
            print("[ResNet50]  ⚠ Gewichte nicht abrufbar – zufällige Initialisierung")
            backbone = models.resnet50(weights=None)

        # Alle Parameter einfrieren
        for param in backbone.parameters():
            param.requires_grad = False

        # Letzte N Blöcke freigeben (Feintuning)
        layer_groups = [backbone.layer1, backbone.layer2,
                        backbone.layer3, backbone.layer4]
        for layer in layer_groups[-unfreeze_blocks:]:
            for param in layer.parameters():
                param.requires_grad = True

        # Alles außer dem FC-Layer übernehmen
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        # backbone.children()[-1] ist der original FC → weglassen

        in_features = backbone.fc.in_features  # = 2048

        # Neuer Klassifikationskopf
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

        # Für Grad-CAM: Referenz auf letzte Conv-Schicht
        self.last_conv = backbone.layer4

    def forward(self, x):
        x = self.backbone(x)
        return self.head(x)

    def get_trainable_params(self):
        """Gibt Anzahl trainierbarer Parameter aus."""
        total  = sum(p.numel() for p in self.parameters())
        train  = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - train
        print(f"[ResNet50]  Gesamt: {total:,}  |  Trainierbar: {train:,}  |  Eingefroren: {frozen:,}")
        return train
