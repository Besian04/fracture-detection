"""
app.py  –  Flask Backend für die Knochenfraktur-Erkennungs-Web-App
==================================================================
Start: python app.py
Dann: http://localhost:5000

Dies ist unser Server, der das Web-Interface bereitstellt und die Vorhersagen (Inferenz)
mithilfe unserer trainierten Deep-Learning-Modelle ausführt.
"""

import base64
import io
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, jsonify, render_template, request
from PIL import Image
from torchvision import transforms

# Hier stellen wir sicher, dass Python unser Projektverzeichnis im Suchpfad findet
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

# Wir importieren unsere eigenen Modellklassen
from models.cnn_model import FractureCNN
from models.transfer_model import FractureResNet

# Flask-App initialisieren. Wir setzen das Template-Verzeichnis direkt auf das Projektverzeichnis.
app = Flask(__name__, template_folder=PROJECT_DIR)

# ---------------------------------------------------------------------------
# Bildvorverarbeitung (Transforms)
# ---------------------------------------------------------------------------
# Jedes hochgeladene Röntgenbild muss exakt so vorverarbeitet werden, wie es beim Training der Fall war.
TRANSFORM = transforms.Compose([
    # Bildgröße auf 224x224 Pixel anpassen (Standardeingangsgröße für ResNet)
    transforms.Resize((224, 224)),
    # Graustufenbilder in 3 RGB-Kanäle umwandeln (erforderlich für ResNet)
    transforms.Grayscale(num_output_channels=3),
    # Konvertiert das Bild in einen PyTorch-Tensor (Skalierung der Pixel auf [0, 1])
    transforms.ToTensor(),
    # Normalisierung mit den Standard-ImageNet-Mitteln und -Standardabweichungen
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Standard-Rechengerät festlegen (hier CPU, da auf dem Webserver meist keine GPU aktiv ist)
DEVICE = torch.device("cpu")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")


# Hilfsfunktion zum sicheren Laden der trainierten Modellgewichte (.pth-Dateien)
def load_model(model_cls, path, **kwargs):
    try:
        # Modellinstanz erzeugen
        m = model_cls(**kwargs)
        # Gewichte laden und auf die CPU mappen
        m.load_state_dict(torch.load(path, map_location=DEVICE))
        # Modell in den Evaluierungsmodus versetzen (deaktiviert Dropout und BatchNorm-Update)
        m.eval()
        print(f"  [OK] {os.path.basename(path)} geladen")
        return m
    except Exception as e:
        # Falls eine Datei fehlt (z. B. vor dem Training), loggen wir das in der Konsole
        print(f"  [ERROR] {path}: {e}")
        return None


# Beim Starten des Servers laden wir direkt beide Modelle in den Speicher
print("\n[App]  Lade Modelle …")
CNN_MODEL    = load_model(FractureCNN,    os.path.join(RESULTS_DIR, "CNN_best.pth"))
RESNET_MODEL = load_model(FractureResNet, os.path.join(RESULTS_DIR, "ResNet50_best.pth"),
                          unfreeze_blocks=0) # Unfreeze=0 da wir hier nur Inferenz machen
print("[App]  Bereit.\n")


# ---------------------------------------------------------------------------
# Grad-CAM (Gradient-weighted Class Activation Mapping)
# ---------------------------------------------------------------------------
# Diese Klasse implementiert die Erklärbarkeits-Methode Grad-CAM.
# Sie zeigt uns als Heatmap, welche Pixel das Modell für seine Entscheidung herangezogen hat.
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        # Wir hängen einen Hooks ein, um die Feature-Maps (Aktivierungen) bei der Vorwärtsleitung abzugreifen
        target_layer.register_forward_hook(
            lambda m, i, o: setattr(self, "activations", o.detach()))
        # Wir hängen einen Hook ein, um die Gradienten (Einfluss der Pixel) beim Backpropagation abzugreifen
        target_layer.register_full_backward_hook(
            lambda m, gi, go: setattr(self, "gradients", go[0].detach()))

    def __call__(self, x):
        self.model.eval()
        # Gradientenberechnung für das Eingabebild aktivieren
        x = x.to(DEVICE).requires_grad_(True)
        # Vorwärtslauf durch das Netz
        logits = self.model(x)

        # Die Rohwahrscheinlichkeit mit Sigmoid berechnen.
        # Da Klasse 0 = 'fractured' und Klasse 1 = 'not fractured' ist (alphabetische Sortierung),
        # liefert der Sigmoid-Wert der Logits direkt die Wahrscheinlichkeit für Klasse 1 ('not fractured').
        raw_prob = torch.sigmoid(logits).item()

        # Die Frakturwahrscheinlichkeit ist somit das Komplement: 1.0 - raw_prob
        fracture_prob = 1.0 - raw_prob

        # Um die Heatmap für 'fractured' (Klasse 0) zu berechnen, müssen wir die Gradienten
        # bezüglich der negierten Logits (-logits) zurückberechnen (Backpropagation).
        self.model.zero_grad()
        (-logits).backward()

        if self.gradients is None or self.activations is None:
            return np.zeros((224, 224)), fracture_prob

        # Die Gradienten über die räumlichen Dimensionen (Breite und Höhe) mitteln, um Gewichte zu erhalten
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        # Gewichtete Summe der Aktivierungen bilden und negative Einflüsse mit ReLU abschneiden
        cam = F.relu((weights * self.activations).sum(1, keepdim=True))
        # Auf die originale Bildgröße von 224x224 Pixeln hochskalieren
        cam = F.interpolate(cam, (224, 224), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        # Min-Max Normalisierung auf den Bereich [0, 1] für eine saubere Farbdarstellung
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)

        return cam, fracture_prob


# Hilfsfunktion, um die jeweils letzte Convolutional-Schicht des gewählten Modells zu holen
def get_cam_layer(model_name):
    if model_name == "cnn" and CNN_MODEL:
        # Beim Custom CNN holen wir die Faltungsschicht des letzten ConvBlocks
        layer = list(CNN_MODEL.features.children())[-1].block[0]
        return GradCAM(CNN_MODEL, layer), CNN_MODEL
    elif model_name == "resnet" and RESNET_MODEL:
        # Beim ResNet50 ist die Referenz auf layer4 bereits in unserer Klasse definiert
        return GradCAM(RESNET_MODEL, RESNET_MODEL.last_conv), RESNET_MODEL
    return None, None


# ---------------------------------------------------------------------------
# Heatmap-Erzeugung und Konvertierung in Base64-String zur Web-Darstellung
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg") # Verhindert GUI-Probleme im Flask-Hintergrundthread
import matplotlib.cm as cm
import matplotlib.pyplot as plt


def heatmap_to_base64(orig_pil, heatmap):
    # 1. Originalbild laden und auf 224x224 skalieren
    orig = np.array(orig_pil.resize((224, 224))).astype(float) / 255.0
    if orig.ndim == 2:
        orig = np.stack([orig] * 3, axis=-1)
    elif orig.shape[2] == 4:
        orig = orig[:, :, :3]

    # 2. Die normalisierte Grad-CAM Heatmap mit Jet-Farbschema (Blau-Rot) einfärben
    colored = cm.jet(heatmap)[..., :3]
    # 3. Überlagerung berechnen (55% Originalbild + 45% Heatmap)
    overlay = (0.55 * orig + 0.45 * colored).clip(0, 1)

    # 4. Eine Abbildung mit 3 Spalten erzeugen: Original | Heatmap | Überlagerung
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    fig.patch.set_facecolor("#0d1117") # Hintergrundfarbe an unser Web-Layout anpassen
    for ax in axes:
        ax.set_facecolor("#0d1117")
        ax.axis("off") # Achsen ausblenden für saubere Medizinoptik

    axes[0].imshow(orig)
    axes[0].set_title("Original", color="white", fontsize=11, pad=6)
    axes[1].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM Heatmap", color="white", fontsize=11, pad=6)
    axes[2].imshow(overlay)
    axes[2].set_title("Überlagerung", color="white", fontsize=11, pad=6)

    # 5. Das geplottete Bild im Speicher als PNG ablegen und in einen Base64-String kodieren
    plt.tight_layout(pad=1.0)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# ---------------------------------------------------------------------------
# Server-Routen
# ---------------------------------------------------------------------------
from flask import send_from_directory


# Hauptseite rendern (liefert unsere index.html aus)
@app.route("/")
def index():
    return render_template("index.html")


# Diese Route liefert die Trainingskurven und Matrizen aus dem results/-Ordner an das Dashboard
@app.route("/results/<path:filename>")
def serve_results(filename):
    return send_from_directory(RESULTS_DIR, filename)


# POST-Schnittstelle zur Bildanalyse
@app.route("/predict", methods=["POST"])
def predict():
    # Prüfen, ob ein Bild hochgeladen wurde
    if "image" not in request.files:
        return jsonify({"error": "Kein Bild hochgeladen"}), 400

    file = request.files["image"]
    model_name = request.form.get("model", "resnet")

    # Versuchen, das Bild mit PIL zu öffnen
    try:
        pil_img = Image.open(file.stream).convert("RGB")
    except Exception:
        return jsonify({"error": "Ungültiges Bildformat"}), 400

    # Bild transformieren und eine Batch-Dimension hinzufügen (Form: 1x3x224x224)
    tensor = TRANSFORM(pil_img).unsqueeze(0)

    # Das gewählte Modell und das entsprechende Grad-CAM-Objekt laden
    grad_cam, model = get_cam_layer(model_name)
    if model is None:
        return jsonify({"error": f"Modell '{model_name}' nicht verfügbar"}), 500

    # Heatmap und Wahrscheinlichkeit berechnen
    heatmap, fracture_prob = grad_cam(tensor)
    # Das Abbildungs-Triptychon erstellen und als Base64-String kodieren
    heatmap_b64 = heatmap_to_base64(pil_img, heatmap)

    # Entscheidungsschwelle bei 50%: Ist die Wahrscheinlichkeit >= 0.5, gilt es als Fraktur
    is_fracture = fracture_prob >= 0.5
    prob_pct = round(fracture_prob * 100, 1)

    # Ergebnis als JSON an das Frontend zurücksenden
    return jsonify({
        "probability": prob_pct,
        "is_fracture": is_fracture,
        "label": "Fraktur erkannt" if is_fracture else "Kein Bruch erkannt",
        "heatmap_b64": heatmap_b64,
        "model_used": "ResNet50" if model_name == "resnet" else "CNN",
    })


# Hauptprogramm-Einstiegspunkt
if __name__ == "__main__":
    # Server im Debug-Modus auf Port 5000 starten
    app.run(debug=True, port=5000)