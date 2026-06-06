"""
app.py  –  Flask Backend für die Knochenfraktur-Erkennungs-Web-App
==================================================================
Start: python app.py
Dann: http://localhost:5000
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

# Projektpfad
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from models.cnn_model import FractureCNN
from models.transfer_model import FractureResNet

app = Flask(__name__, template_folder=PROJECT_DIR)

# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ---------------------------------------------------------------------------
# Klassen-Mapping prüfen
# Klasse 0 = fractured, Klasse 1 = not fractured (alphabetisch)
# → sigmoid output nahe 0 = fractured, nahe 1 = not fractured
# → Fraktur-Wahrscheinlichkeit = 1 - sigmoid(logit)
# ---------------------------------------------------------------------------
DEVICE = torch.device("cpu")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")


def load_model(model_cls, path, **kwargs):
    try:
        m = model_cls(**kwargs)
        m.load_state_dict(torch.load(path, map_location=DEVICE))
        m.eval()
        print(f"  ✓ {os.path.basename(path)} geladen")
        return m
    except Exception as e:
        print(f"  ✗ {path}: {e}")
        return None


print("\n[App]  Lade Modelle …")
CNN_MODEL    = load_model(FractureCNN,    os.path.join(RESULTS_DIR, "CNN_best.pth"))
RESNET_MODEL = load_model(FractureResNet, os.path.join(RESULTS_DIR, "ResNet50_best.pth"),
                          unfreeze_blocks=0)
print("[App]  Bereit.\n")


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(
            lambda m, i, o: setattr(self, "activations", o.detach()))
        target_layer.register_full_backward_hook(
            lambda m, gi, go: setattr(self, "gradients", go[0].detach()))

    def __call__(self, x):
        self.model.eval()
        x = x.to(DEVICE).requires_grad_(True)
        logits = self.model(x)

        # sigmoid output: nahe 0 = fractured (Klasse 0), nahe 1 = not fractured (Klasse 1)
        raw_prob = torch.sigmoid(logits).item()

        # Fraktur-Wahrscheinlichkeit (was wir anzeigen wollen)
        fracture_prob = 1.0 - raw_prob

        # Backward auf logit (negieren wenn fractured = Klasse 0)
        # Um Grad-CAM für "fractured" zu berechnen, backpropagate -logits
        self.model.zero_grad()
        (-logits).backward()  # negieren → Gradienten für Klasse 0 (fractured)

        if self.gradients is None or self.activations is None:
            return np.zeros((224, 224)), fracture_prob

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(1, keepdim=True))
        cam = F.interpolate(cam, (224, 224), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)

        return cam, fracture_prob


def get_cam_layer(model_name):
    if model_name == "cnn" and CNN_MODEL:
        layer = list(CNN_MODEL.features.children())[-1].block[0]
        return GradCAM(CNN_MODEL, layer), CNN_MODEL
    elif model_name == "resnet" and RESNET_MODEL:
        return GradCAM(RESNET_MODEL, RESNET_MODEL.last_conv), RESNET_MODEL
    return None, None


# ---------------------------------------------------------------------------
# Heatmap → PNG Base64
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt


def heatmap_to_base64(orig_pil, heatmap):
    orig = np.array(orig_pil.resize((224, 224))).astype(float) / 255.0
    if orig.ndim == 2:
        orig = np.stack([orig] * 3, axis=-1)
    elif orig.shape[2] == 4:
        orig = orig[:, :, :3]

    colored = cm.jet(heatmap)[..., :3]
    overlay = (0.55 * orig + 0.45 * colored).clip(0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    fig.patch.set_facecolor("#0d1117")
    for ax in axes:
        ax.set_facecolor("#0d1117")
        ax.axis("off")

    axes[0].imshow(orig)
    axes[0].set_title("Original", color="white", fontsize=11, pad=6)
    axes[1].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM Heatmap", color="white", fontsize=11, pad=6)
    axes[2].imshow(overlay)
    axes[2].set_title("Überlagerung", color="white", fontsize=11, pad=6)

    plt.tight_layout(pad=1.0)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Kein Bild hochgeladen"}), 400

    file = request.files["image"]
    model_name = request.form.get("model", "resnet")

    try:
        pil_img = Image.open(file.stream).convert("RGB")
    except Exception:
        return jsonify({"error": "Ungültiges Bildformat"}), 400

    tensor = TRANSFORM(pil_img).unsqueeze(0)

    grad_cam, model = get_cam_layer(model_name)
    if model is None:
        return jsonify({"error": f"Modell '{model_name}' nicht verfügbar"}), 500

    heatmap, fracture_prob = grad_cam(tensor)
    heatmap_b64 = heatmap_to_base64(pil_img, heatmap)

    is_fracture = fracture_prob >= 0.5
    prob_pct = round(fracture_prob * 100, 1)

    return jsonify({
        "probability": prob_pct,
        "is_fracture": is_fracture,
        "label": "Fraktur erkannt" if is_fracture else "Kein Bruch erkannt",
        "heatmap_b64": heatmap_b64,
        "model_used": "ResNet50" if model_name == "resnet" else "CNN",
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)