"""
cnn_model.py
------------
Einfaches CNN als Basismodell (Modell 1).

Architektur:
    3 × ConvBlock (Conv → BN → ReLU → MaxPool)
    GlobalAvgPool → FC-Schichten → Sigmoid-Ausgabe
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv2d → BatchNorm → ReLU → MaxPool"""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, pool: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class FractureCNN(nn.Module):
    """
    Einfaches CNN für binäre Klassifikation (Fraktur / kein Bruch).

    Input:  (B, 3, 224, 224)
    Output: (B, 1)  → Sigmoid-Wahrscheinlichkeit
    """

    def __init__(self, dropout: float = 0.5):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(3,   32),   # → 112×112
            ConvBlock(32,  64),   # →  56×56
            ConvBlock(64, 128),   # →  28×28
            ConvBlock(128, 256),  # →  14×14
        )

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # → (B, 256, 1, 1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.global_avg_pool(x)
        return self.classifier(x)          # logits (kein Sigmoid → BCEWithLogits)

    def get_feature_maps(self, x):
        """Gibt Feature-Maps der letzten Conv-Schicht zurück (für Grad-CAM)."""
        return self.features(x)
