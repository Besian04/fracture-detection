# transfer_model.py
# -----------------
# Unser fortschrittlicheres Modell 2 auf Basis von Transfer Learning.
# Wir verwenden das bewährte ResNet50-Modell, das auf ImageNet vortrainiert wurde.
# Dadurch müssen wir nicht bei Null anfangen, sondern nutzen bereits gelernte visuelle Konzepte.

import torch
import torch.nn as nn
from torchvision import models


class FractureResNet(nn.Module):
    """
    ResNet50 mit einem für unsere Zwecke angepassten Klassifikationskopf.
    Input: Ein Batch von Bildern mit (Batchgröße, 3 Farbkanäle, 224x224 Pixel)
    Output: Logits für das binäre Klassifikationsproblem (Fraktur/gesund).
    """

    def __init__(self, dropout: float = 0.4, unfreeze_blocks: int = 2):
        """
        Args:
            dropout:         Dropout-Rate im neuen Klassifikationskopf (gegen Overfitting).
            unfreeze_blocks: Anzahl der letzten ResNet-Layer-Gruppen, die wir beim Feintuning
                             mit-trainieren wollen (0 = nur Kopf trainieren, max 4).
        """
        super().__init__()

        # Wir versuchen, das vortrainierte ResNet50-Modell von PyTorch zu laden
        try:
            # Wir verwenden die neuesten verfügbaren ImageNet-Gewichte
            weights = models.ResNet50_Weights.IMAGENET1K_V2
            backbone = models.resnet50(weights=weights)
            print("[ResNet50]  Vortrainierte ImageNet-Gewichte geladen [OK]")
        except Exception:
            # Falls keine Internetverbindung da ist, initialisieren wir das Modell zufällig
            print("[ResNet50]  WARNUNG: Gewichte nicht abrufbar - zufaellige Initialisierung")
            backbone = models.resnet50(weights=None)

        # Schritt 1: Erstmal frieren wir alle Gewichte des vortrainierten Backbones ein
        for param in backbone.parameters():
            param.requires_grad = False

        # Schritt 2: Nun geben wir die letzten N Faltungsblöcke wieder frei (Feintuning)
        # Dadurch kann sich das Modell besser an die Besonderheiten von Röntgenbildern anpassen.
        layer_groups = [backbone.layer1, backbone.layer2,
                        backbone.layer3, backbone.layer4]
        for layer in layer_groups[-unfreeze_blocks:]:
            for param in layer.parameters():
                param.requires_grad = True

        # Schritt 3: Wir entfernen die letzte, originale vollvernetzte Schicht (FC) von ResNet,
        # da diese für 1000 ImageNet-Klassen ausgelegt ist, und speichern den Rest als Sequential-Modell.
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        # Wir lesen die Anzahl der Ausgangsmerkmale des Backbones ab (beim ResNet50 sind das 2048)
        in_features = backbone.fc.in_features

        # Schritt 4: Wir bauen unseren eigenen, neuen Klassifikationskopf für das Binärproblem.
        self.head = nn.Sequential(
            # Flatten konvertiert das Signal von (B, 2048, 1, 1) in (B, 2048)
            nn.Flatten(),
            # Erste vollvernetzte Schicht: 2048 Eingänge -> 256 Zwischenneuronen
            nn.Linear(in_features, 256),
            # Aktivierungsfunktion ReLU
            nn.ReLU(inplace=True),
            # Dropout zur Regularisierung während des Trainings
            nn.Dropout(dropout),
            # Ausgangsschicht: 256 -> 1 Ausgang (Logit für Brucherkennung)
            nn.Linear(256, 1),
        )

        # Referenz auf die letzte Faltungsschicht speichern - das ist extrem wichtig für Grad-CAM
        self.last_conv = backbone.layer4

    def forward(self, x):
        # 1. Das Röntgenbild durch das ResNet50-Rückgrat schicken
        x = self.backbone(x)
        # 2. Durch unseren angepassten Klassifikationskopf leiten und Logits erhalten
        return self.head(x)

    def get_trainable_params(self):
        """
        Gibt die Anzahl der trainierbaren vs. eingefrorenen Parameter in der Konsole aus.
        Sehr nützlich, um zu überprüfen, ob das Unfreezing korrekt geklappt hat.
        """
        total  = sum(p.numel() for p in self.parameters())
        train  = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - train
        print(f"[ResNet50]  Gesamt: {total:,}  |  Trainierbar: {train:,}  |  Eingefroren: {frozen:,}")
        return train
