# gradcam.py
# ----------
# Hier implementieren wir Grad-CAM (Gradient-weighted Class Activation Mapping).
# Diese Methode visualisiert die internen Entscheidungswege unseres Netzes.
# Sie erzeugt farbige Heatmaps, um zu beweisen, worauf das Modell beim Röntgenbild geschaut hat.

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from torchvision import transforms

# Hilfs-Transformation zur Bildvorbereitung für die Validierung
VAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Denormalisierung: Um das normalisierte Bild wieder als Standardbild anzeigen zu können,
# machen wir die Normalisierung mit den mathematisch inversen Formeln rückgängig.
DENORM = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)


class GradCAM:
    """
    Diese Klasse führt die Grad-CAM-Berechnung aus. Sie funktioniert modular
    für jedes PyTorch-Modell und greift auf eine gewünschte Ziel-Faltungsschicht zu.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model        = model            # Das neuronale Netzwerk
        self.target_layer = target_layer     # Die Convolutional-Schicht, deren Feature-Maps wir analysieren
        self.gradients    = None             # Speichert die Gradienten aus dem Backward Pass
        self.activations  = None             # Speichert die Aktivierungs-Werte aus dem Forward Pass
        self._hooks       = []               # Liste der registrierten Hooks
        self._register_hooks()               # Hooks beim Initialisieren registrieren

    def _register_hooks(self):
        # Forward Hook: Fängt die Aktivierungen (Feature-Maps) der Ziel-Schicht ab
        def forward_hook(module, input, output):
            self.activations = output.detach()

        # Backward Hook: Fängt die Gradienten (Fehlersignale) ab, die zur Ziel-Schicht fließen
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        # Hooks im PyTorch-Modell registrieren
        self._hooks.append(
            self.target_layer.register_forward_hook(forward_hook)
        )
        self._hooks.append(
            self.target_layer.register_full_backward_hook(backward_hook)
        )

    def remove_hooks(self):
        # Entfernt die Hooks nach der Analyse wieder, damit das normale Training nicht blockiert wird
        for h in self._hooks:
            h.remove()

    def __call__(self, image_tensor: torch.Tensor) -> np.ndarray:
        """
        Berechnet die 2D Grad-CAM Heatmap für ein übergebenes Röntgenbild.
        """
        self.model.eval()
        device = next(self.model.parameters()).device
        x      = image_tensor.to(device)
        # Gradientenberechnung für das Eingabebild aktivieren
        x.requires_grad_(True)

        # 1. Vorwärtslauf durchführen (Aktivierungen werden per Hook abgefangen)
        logits = self.model(x)
        prob   = torch.sigmoid(logits)

        # 2. Rückwärtslauf durchführen (Gradienten werden per Hook abgefangen)
        self.model.zero_grad()
        logits.backward()

        # 3. Grad-CAM Formel anwenden:
        # A) Gewichte berechnen, indem wir die Gradienten global über die Pixel mitteln (Global Average Pooling)
        weights  = self.gradients.mean(dim=(2, 3), keepdim=True)
        # B) Linearkombination der Feature-Maps mit diesen Gewichten bilden
        cam      = (weights * self.activations).sum(dim=1, keepdim=True)
        # C) ReLU anwenden (uns interessieren nur Merkmale, die positiv zur Bruch-Entscheidung beitragen)
        cam      = F.relu(cam)

        # 4. Die Heatmap wieder auf die volle Bildgröße (224x224) skalieren
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # 5. Normalisieren, damit die Intensitätswerte sauber zwischen 0.0 und 1.0 liegen
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)

        return cam, float(prob.item())


def visualize_gradcam(model, target_layer, image_path: str = None,
                      image_tensor: torch.Tensor = None,
                      save_path: str = None,
                      class_names=("No Fracture", "Fracture"),
                      title: str = "Grad-CAM"):
    """
    Erzeugt und visualisiert die Grad-CAM Heatmap für ein einzelnes Bild.
    Es wird das Originalbild, die Heatmap und die Farbüberlagerung nebeneinander geplottet.
    """
    # Bild entweder aus Pfad laden oder direkt als Tensor übernehmen
    if image_path is not None:
        pil_img = Image.open(image_path).convert("RGB")
        tensor  = VAL_TRANSFORMS(pil_img).unsqueeze(0)
    elif image_tensor is not None:
        tensor  = image_tensor.unsqueeze(0) if image_tensor.dim() == 3 else image_tensor
        # Für die Visualisierung denormalisieren wir den Tensor
        pil_arr = DENORM(tensor.squeeze()).permute(1, 2, 0).clamp(0, 1).numpy()
        pil_img = Image.fromarray((pil_arr * 255).astype(np.uint8))
    else:
        raise ValueError("image_path oder image_tensor muss angegeben werden.")

    # Grad-CAM berechnen
    grad_cam   = GradCAM(model, target_layer)
    heatmap, prob = grad_cam(tensor)
    grad_cam.remove_hooks() # Hooks wieder freigeben

    pred_class = 1 if prob >= 0.5 else 0
    pred_label = class_names[pred_class]

    orig_arr = np.array(pil_img.resize((224, 224))) / 255.0

    # 1. Jet-Farbschema (Blau = kalt, Rot = heiß) auf die 2D-Heatmap anwenden
    colored_heatmap = cm.jet(heatmap)[..., :3]
    # 2. 50% Originalbild und 50% Heatmap überlagern (Alpha-Blending)
    overlay         = 0.5 * orig_arr + 0.5 * colored_heatmap

    # 3. Drei Spalten nebeneinander zeichnen
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle(f"{title}  |  Vorhersage: {pred_label} ({prob:.1%})",
                 fontsize=13, fontweight="bold")

    axes[0].imshow(orig_arr, cmap="gray" if orig_arr.ndim == 2 else None)
    axes[0].set_title("Original Röntgenbild")
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")
    plt.colorbar(cm.ScalarMappable(cmap="jet"), ax=axes[1], fraction=0.046)

    axes[2].imshow(overlay.clip(0, 1))
    axes[2].set_title("Überlagerung")
    axes[2].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Grad-CAM gespeichert: {save_path}")
    plt.show()
    return heatmap, prob


def batch_gradcam(model, target_layer, val_loader, device,
                  n_samples: int = 6, class_names=("No Fracture", "Fracture"),
                  save_path: str = None):
    """
    Diese Funktion erstellt ein großes Übersichtsraster mit Grad-CAM-Analysen für mehrere
    Bilder aus dem Validierungsset. Das nutzen wir im Bericht und im Modell-Dashboard.
    """
    model.eval()
    images_list, labels_list = [], []

    # Wir ziehen uns die ersten n Bilder aus dem Dataloader
    for images, labels in val_loader:
        images_list.append(images)
        labels_list.append(labels)
        if sum(len(b) for b in images_list) >= n_samples:
            break

    images = torch.cat(images_list)[:n_samples]
    labels = torch.cat(labels_list)[:n_samples]

    grad_cam = GradCAM(model, target_layer)

    # Raster erzeugen (Anzahl der Bilder x 3 Spalten)
    fig, axes = plt.subplots(n_samples, 3, figsize=(12, n_samples * 3.5))
    fig.suptitle("Grad-CAM Batch-Visualisierung", fontsize=14, fontweight="bold")

    for i in range(n_samples):
        img_t  = images[i]
        # Grad-CAM für das Bild berechnen
        heatmap, prob = grad_cam(img_t.unsqueeze(0))

        # Bild wieder für den Bildschirm aufbereiten (denormalisieren)
        orig = DENORM(img_t).permute(1, 2, 0).clamp(0, 1).numpy()
        colored = cm.jet(heatmap)[..., :3]
        overlay = (0.5 * orig + 0.5 * colored).clip(0, 1)

        pred  = 1 if prob >= 0.5 else 0
        truth = int(labels[i].item())
        # Titel grün färben bei korrekter Vorhersage, rot bei Fehlern
        color = "green" if pred == truth else "red"

        # Spalte 1: Originalbild mit Ground-Truth Label
        axes[i, 0].imshow(orig, cmap="gray")
        axes[i, 0].set_title(f"GT: {class_names[truth]}", color=color, fontsize=9)
        axes[i, 0].axis("off")

        # Spalte 2: Reine Heatmap
        axes[i, 1].imshow(heatmap, cmap="jet")
        axes[i, 1].set_title(f"Pred: {class_names[pred]} ({prob:.0%})", fontsize=9)
        axes[i, 1].axis("off")

        # Spalte 3: Farbiges Overlay
        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title("Overlay", fontsize=9)
        axes[i, 2].axis("off")

    grad_cam.remove_hooks() # Hooks nach getaner Arbeit aufräumen
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=130, bbox_inches="tight")
        print(f"  Batch Grad-CAM gespeichert: {save_path}")
    plt.show()