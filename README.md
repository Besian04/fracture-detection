# Knochenfraktur-Erkennung mit Deep Learning

Automatische Erkennung von Knochenfrakturen in Röntgenbildern mittels CNN und Transfer Learning.

---

Für Das Clonen:

```
git clone https://github.com/DEIN-USERNAME/fracture_detection.git

cd fracture_detection

# Setup ausführen

bash setup.sh

# Virtuelle Umgebung aktivieren

source .venv/bin/activate

# Web-App starten

python app.py
```

---

## Projektstruktur

```
fracture_detection/
├── main.py                   # Hauptskript (gesamte Pipeline)
├── requirements.txt
├── models/
│   ├── cnn_model.py          # Modell 1: Custom CNN (Baseline)
│   └── transfer_model.py     # Modell 2: ResNet50 Transfer Learning
└── utils/
    ├── dataset.py            # Datensatz-Loader & Augmentation
    ├── trainer.py            # Trainingsschleife + Early Stopping
    ├── evaluation.py         # Metriken, Confusion Matrix, ROC
    ├── gradcam.py            # Grad-CAM Explainability
    └── compare_models.py     # Modellvergleich & Visualisierung
```

---

## Schnellstart

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Dataset herunterladen

**Option A** (empfohlen für den Einstieg):
https://www.kaggle.com/datasets/foyez767/x-ray-images-of-fractured-and-healthy-bones

**Option B** (größer, anatomisch diverser):
https://www.kaggle.com/datasets/bmadushanirodrigo/fracture-multi-region-x-ray-data

Kaggle CLI:
```bash
kaggle datasets download foyez767/x-ray-images-of-fractured-and-healthy-bones
unzip *.zip -d data/
```

### 3. Erwartete Ordnerstruktur

**Variante A** (mit vorgegebenem Train/Val-Split):
```
data/
├── train/
│   ├── fractured/
│   └── not fractured/
└── val/
    ├── fractured/
    └── not fractured/
```

**Variante B** (automatischer 80/20 Split):
```
data/
├── fractured/
└── not fractured/
```

### 4. Training starten

```bash
# Beide Modelle trainieren (Standard)
python main.py --data_dir ./data --epochs 30

# Nur ResNet50 trainieren
python main.py --data_dir ./data --skip_cnn

# Nur CNN trainieren
python main.py --data_dir ./data --skip_resnet

# Alle Optionen
python main.py \
  --data_dir ./data \
  --results_dir ./results \
  --epochs 40 \
  --batch_size 32 \
  --gradcam_samples 8
```

---

## Modelle

### Modell 1: Custom CNN

Einfaches Convolutional Neural Network als Basismodell:
- 4 ConvBlöcke (Conv → BatchNorm → ReLU → MaxPool)
- Global Average Pooling
- FC-Klassifikationskopf mit Dropout
- ~2,5M Parameter

### Modell 2: ResNet50 Transfer Learning

Vortrainiertes ResNet50 (ImageNet), angepasst für Frakturerkennung:
- Alle Schichten eingefroren außer Layer3/Layer4 + neuem Head
- Neuer Klassifikationskopf: Linear(2048→256) → Dropout → Linear(256→1)
- ~25M Parameter, davon ~10M trainierbar

---

## Metriken

| Metrik         | Beschreibung                                      |
|----------------|---------------------------------------------------|
| Accuracy       | Anteil korrekt klassifizierter Bilder             |
| Precision      | Anteil echter Frakturen unter positiven Preds.    |
| **Recall**     | Anteil erkannter Frakturen (kritisch!)            |
| F1-Score       | Harmonisches Mittel aus Precision & Recall        |
| AUC-ROC        | Fläche unter der ROC-Kurve                        |

> **Recall** ist die wichtigste Metrik: Eine übersehene Fraktur ist gefährlicher als ein Falsch-Positiv.

---

## Ausgabe-Artefakte (results/)

| Datei                          | Inhalt                                      |
|--------------------------------|---------------------------------------------|
| `CNN_best.pth`                 | Bestes CNN-Checkpoint                       |
| `ResNet50_best.pth`            | Bestes ResNet50-Checkpoint                  |
| `CNN_training_curves.png`      | Loss & Accuracy Lernkurven (CNN)            |
| `ResNet50_training_curves.png` | Loss & Accuracy Lernkurven (ResNet50)       |
| `CNN_confusion_matrix.png`     | Confusion Matrix (CNN)                      |
| `ResNet50_confusion_matrix.png`| Confusion Matrix (ResNet50)                 |
| `CNN_roc_curve.png`            | ROC-Kurve (CNN)                             |
| `ResNet50_roc_curve.png`       | ROC-Kurve (ResNet50)                        |
| `CNN_gradcam.png`              | Grad-CAM Batch-Visualisierung (CNN)         |
| `ResNet50_gradcam.png`         | Grad-CAM Batch-Visualisierung (ResNet50)    |
| `model_comparison.png`         | Direkter Metrik-Vergleich beider Modelle    |

---

## Erklärbarkeit: Grad-CAM

Grad-CAM erzeugt Heatmaps, die zeigen, auf welche Bildbereiche das Modell bei seiner Entscheidung geachtet hat. So kann überprüft werden, ob das Modell tatsächlich die Knochen analysiert oder sich auf irrelevante Bereiche konzentriert.

```
Original Röntgenbild → Grad-CAM Heatmap → Überlagerung
```

---

## Hyperparameter

| Parameter       | CNN     | ResNet50 |
|-----------------|---------|----------|
| Learning Rate   | 1e-3    | 3e-4     |
| Batch Size      | 32      | 32       |
| Dropout         | 0.5     | 0.4      |
| pos_weight      | 1.5     | 1.5      |
| Early Stopping  | 7 Epochen | 7 Epochen |
| LR Scheduler    | ReduceLROnPlateau | ReduceLROnPlateau |

---

## Technologien

- **PyTorch** – Deep Learning Framework
- **torchvision** – Transfer Learning (ResNet50), Transforms
- **scikit-learn** – Metriken, Confusion Matrix
- **matplotlib** – Visualisierungen
- **Pillow** – Bildverarbeitung
