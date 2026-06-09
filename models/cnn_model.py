# cnn_model.py

# Hier definieren wir unser erstes, selbstgebautes CNN-Modell (Modell 1).
# Es dient als Baseline-Modell, um eine solide Grundlage für den Vergleich zu haben.
# Wir setzen auf eine klassische Struktur: Faltungsblöcke zur Feature-Extraktion,
# gefolgt von einer Klassifikationsschicht.

import torch
import torch.nn as nn


# Dieser Block fasst die typische Abfolge für Faltungsnetzwerke zusammen, damit wir Code sparen.
class ConvBlock(nn.Module):
    """
    Wir bauen einen Faltungsblock bestehend aus:
    Faltung (Conv2d) -> Batch-Normalisierung (BatchNorm2d) -> Aktivierung (ReLU) -> Pooling (MaxPool2d)
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, pool: bool = True):
        super().__init__()
        # Wir listen die Schichten nacheinander in einem Array auf
        layers = [
            # Faltungsschicht: sucht nach Mustern. padding=1 erhält die Bildauflösung, bias=False da BatchNorm folgt.
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=1, bias=False),
            # Batch-Normalisierung: stabilisiert und beschleunigt das Training durch Skalierung der Aktivierungen
            nn.BatchNorm2d(out_channels),
            # Aktivierungsfunktion: bringt Nichtlinearität ins Modell, damit es komplexe Muster lernen kann
            nn.ReLU(inplace=True),
        ]
        # Falls Pooling gewünscht ist, fügen wir Max-Pooling hinzu (halbiert die Bildgröße)
        if pool:
            layers.append(nn.MaxPool2d(2))
        # Wir fassen alle Schichten zu einem sequentiellen Block zusammen
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        # Das Signal wird einfach durch den Block durchgereicht
        return self.block(x)


# Unser Hauptnetzwerk zur Erkennung von Brüchen
class FractureCNN(nn.Module):
    """
    Unser Custom-CNN für die binäre Klassifikation (Fraktur oder gesund).
    Input: Ein Batch von Bildern mit (Batchgröße, 3 Farbkanäle, 224x224 Pixel)
    Output: Ein Logit-Wert pro Bild. Höhere Werte bedeuten 'gesund', niedrigere 'Fraktur'
    """

    def __init__(self, dropout: float = 0.5):
        super().__init__()

        # Der Feature-Extractor: Hier suchen wir in 4 Stufen nach Kanten, Formen und Texturen.
        # Nach jedem Block wird die Auflösung halbiert und die Anzahl der Feature-Channels verdoppelt.
        self.features = nn.Sequential(
            ConvBlock(3,   32),   # Input: 3 Channels (RGB) -> Output: 32 Channels. Auflösung: 224x224 -> 112x112
            ConvBlock(32,  64),   # Input: 32 -> Output: 64. Auflösung: 112x112 -> 56x56
            ConvBlock(64, 128),   # Input: 64 -> Output: 128. Auflösung: 56x56 -> 28x28
            ConvBlock(128, 256),  # Input: 128 -> Output: 256. Auflösung: 28x28 -> 14x14
        )

        # Global Average Pooling: Reduziert die 14x14-Feature-Maps auf einen einzigen Durchschnittswert pro Channel.
        # Dadurch wird das Modell unabhängig von der genauen Position eines Merkmals und spart massig Parameter.
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # Output-Form: (Batchgröße, 256, 1, 1)

        # Unser Klassifikationskopf: Er berechnet aus den extrahierten Merkmalen die endgültige Vorhersage.
        self.classifier = nn.Sequential(
            # Wandelt das 4D-Tensor-Format (B, 256, 1, 1) in ein flaches 2D-Format (B, 256) um
            nn.Flatten(),
            # Lineare Schicht zur Kombination der Features: 256 Eingänge -> 128 Neuronen
            nn.Linear(256, 128),
            # ReLU-Aktivierung für die Zwischenschicht
            nn.ReLU(inplace=True),
            # Dropout: Deaktiviert im Training zufällig Neuronen, um Overfitting vorzubeugen
            nn.Dropout(dropout),
            # Endschicht: 128 Eingänge -> 1 Ausgang (Logit für die Klassifikation)
            nn.Linear(128, 1),
        )

    def forward(self, x):
        # 1. Feature-Extraktion durchführen
        x = self.features(x)
        # 2. Räumliche Dimensionen auf 1x1 reduzieren
        x = self.global_avg_pool(x)
        # 3. Klassifikation berechnen und Logits zurückgeben
        return self.classifier(x)

    def get_feature_maps(self, x):
        """
        Diese Methode brauchen wir für Grad-CAM, um die Aktivierungen der letzten
        Faltungsschicht abzugreifen und als Heatmap darzustellen.
        """
        return self.features(x)
